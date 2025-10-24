#!/usr/bin/env python3
"""
DSY-RS drive (Modbus RTU over USB-RS485):
- Position mode
- Internal multi-segment position (incremental displacements)
- 1 unit = 1 degree (P04.05 = 360)
- Segment 1: +180° move relative to current, dwell, then −180° back
- Works across pymodbus 3.x variants by adapting unit/slave handling.

If anything fails, a clear exception is printed.
"""

import argparse
import time, sys, inspect
from pymodbus.client import ModbusSerialClient
from pymodbus.pdu import ExceptionResponse
import pymodbus

# ---------- USER CONFIG ----------
PORT    = "/dev/ttyUSB0"
BAUD    = 115200
SLAVEID = 1          # Drive address (P10.00). Your mbpoll used -a 1.
ACC_MS  = 200        # Acc/Dec basis (ms to 1000 rpm)
MAX_RPM = 300        # Peak speed
DWELL_MS = 500       # Pause at 180°
# Fixed units-per-rev - was originally 10000
UNITS_PER_REV = 10000
# How much to move
MOVE_DEG = 180
# ---------------------------------

# ----- Register addresses (decimal): addr = group*256 + index -----
P0000 = 0                # P00.00 Control mode (0: position)
P0400 = 1024             # P04.00 Position command source (4: internal multi-segment)
P0405 = 1029             # P04.05 Units per rev (32-bit)

# Multi-segment parameters (segment 1)
P1308 = 3336             # P13.08 Displacement (32-bit signed, hi/lo)
P1310 = 3338             # P13.10 Max speed (rpm)
P1311 = 3339             # P13.11 Acc/Dec basis (ms to 1000 rpm)
P1312 = 3340             # P13.12 Wait after positioning (ms)
P1300 = 3328             # P13.00 Mode select
P1301 = 3329             # P13.01 Start segment
P1302 = 3330             # P13.02 End segment
P1303 = 3331             # P13.03 Residual handling
P1305 = 3333             # P13.05 Position control method (0 incremental)

# DI mapping: DI1 (S-ON), DI2 (PSEC_EN)
P0201 = 513              # P02.01 DI1 function code
P0202 = 514              # P02.02 DI2 function code
FUNIN_SERVO_ON = 1
FUNIN_PSEC_EN = 29

# Forced DI (software toggle)
P1110 = 2826             # P11.10 Forced DI enable (1=enable)
P1111 = 2827             # P11.11 Forced DI mask (bit0=DI1, bit1=DI2,...)
P1101 = 2817             # P11.01 Fault reset
DI1_MASK = 0x0001
DI2_MASK = 0x0002

# Optional sanity reads
P1000 = 2560             # P10.00 (comm address)
P1801 = 4609             # P18.01 (speed feedback)
P1807 = 4615             # P18.07 (absolute position, 32-bit signed)
P0901 = 2305             # P09.01 (fault / error code register, 16-bit)
def die(msg): print(msg, file=sys.stderr); sys.exit(1)

def make_client():
    # Create a basic serial client (no unit_id in ctor for 3.11.x)
    return ModbusSerialClient(
        port=PORT,
        baudrate=BAUD,
        bytesize=8,
        parity='N',   # 8N1 to match your mbpoll
        stopbits=1,
        timeout=1.0,
    )

def detect_unit_strategy(client, desired_id):
    """
    Return a dict with:
      - call_kw: 'slave' or 'unit' or None (if not supported per-call)
      - set_attr: attribute name to set on client (e.g., 'unit_id', 'slave_id', etc.) or None
    Then apply the attribute if available.
    """
    # 1) See what read_holding_registers accepts
    sig = inspect.signature(client.read_holding_registers)
    params = sig.parameters
    call_kw = None
    if 'slave' in params: call_kw = 'slave'
    elif 'unit' in params: call_kw = 'unit'

    # 2) If no per-call kw, see if client has a settable attribute
    set_attr = None
    for cand in ('unit_id', 'slave_id', 'unit', 'slave', 'address'):
        try:
            if hasattr(client, cand):
                setattr(client, cand, desired_id)
                # read back to confirm attribute exists
                getattr(client, cand)
                set_attr = cand
                break
        except Exception:
            pass

    return {'call_kw': call_kw, 'set_attr': set_attr}

def chk(resp):
    if hasattr(resp, "isError") and resp.isError():
        raise RuntimeError(f"Modbus error: {resp}")
    return resp

def writer16(client, addr, val, call_kw):
    kw = {call_kw: SLAVEID} if call_kw else {}
    resp = client.write_register(address=addr, value=val, **kw)
    if hasattr(resp, "isError") and resp.isError():
        if isinstance(resp, ExceptionResponse) and resp.function_code == 0x86:
            alt = client.write_registers(address=addr, values=[val], **kw)
            chk(alt)
            return
        raise RuntimeError(f"Modbus error: {resp}")
    return resp

def writer32s(client, addr, sval, call_kw, *, word_order="lohi"):
    if sval < 0: sval = (1 << 32) + sval
    hi = (sval >> 16) & 0xFFFF
    lo = sval & 0xFFFF
    if word_order == "hilo":
        payload = [hi, lo]
    else:
        payload = [lo, hi]
    if call_kw:
        chk(client.write_registers(address=addr, values=payload, **{call_kw: SLAVEID}))
    else:
        chk(client.write_registers(address=addr, values=payload))

def reader16(client, addr, n, call_kw):
    if call_kw:
        rr = chk(client.read_holding_registers(address=addr, count=n, **{call_kw: SLAVEID}))
    else:
        rr = chk(client.read_holding_registers(address=addr, count=n))
    return rr.registers

def reader32s(client, addr, call_kw, *, word_order="lohi"):
    regs = reader16(client, addr, 2, call_kw)
    if word_order == "hilo":
        hi, lo = regs
    else:
        lo, hi = regs
    value = (hi << 16) | lo
    if value & 0x80000000:
        value -= 0x100000000
    return value

def deg_to_units(deg, units_per_rev=UNITS_PER_REV):

    return int(round((units_per_rev * deg) / 360.0))

def units_to_deg(units, units_per_rev=UNITS_PER_REV):
    return (units / units_per_rev) * 360.0

def bcd16_to_int(value):
    digits = [(value >> shift) & 0xF for shift in (12, 8, 4, 0)]
    if any(d > 9 for d in digits):
        return None
    return digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]

def read_encoder_state(client, call_kw, units_per_rev):
    pos_units = reader32s(client, P1807, call_kw)
    pos_deg = units_to_deg(pos_units, units_per_rev)
    print(f"Encoder position: {pos_units} units ({pos_deg:.2f}°)")
    return pos_units, pos_deg

def read_error_code(client, call_kw):
    try:
        raw = reader16(client, P0901, 1, call_kw)
    except Exception as exc:
        print(f"Failed to read P09.01 error code: {exc}")
        return None
    if not raw:
        print("P09.01 read returned no registers.")
        return None
    value = raw[0]
    signed = value if value < 0x8000 else value - 0x10000
    bcd = bcd16_to_int(value)
    print(f"P09.01 error code raw: 0x{value:04X}")
    print(f"P09.01 error code unsigned: {value}")
    print(f"P09.01 error code signed: {signed}")
    if bcd is not None:
        print(f"P09.01 error code BCD: {bcd:04d}")
    return {
        "raw": value,
        "unsigned": value,
        "signed": signed,
        "bcd": bcd,
    }

def set_psec_level(client, call_kw, base_mask, enable):
    mask = base_mask | (DI2_MASK if enable else 0)
    writer16(client, P1111, mask, call_kw)
    return mask

def run_profile():
    print("pymodbus version:", pymodbus.__version__)
    client = make_client()
    if not client.connect():
        die(f"Failed to open {PORT} at {BAUD} bps")

    strat = {'call_kw': None}
    base_mask = 0
    forced_di_active = False
    try:
        strat = detect_unit_strategy(client, SLAVEID)
        print(f"Per-call kw: {strat['call_kw']}, client attr set: {strat['set_attr']}")

        # Optional sanity read of P10.00
        try:
            addr_echo = reader16(client, P1000, 1, strat['call_kw'])[0]
            print(f"P10.00 (drive address) reads: {addr_echo}")
        except Exception as e:
            print(f"Warning: could not read P10.00: {e}")

        # ---- Map DI1 -> S-ON, DI2 -> PSEC_EN ----
        writer16(client, P0201, FUNIN_SERVO_ON, strat['call_kw'])
        writer16(client, P0202, FUNIN_PSEC_EN, strat['call_kw'])

        # ---- Force DI control (start with servo OFF) ----
        writer16(client, P1110, 1, strat['call_kw'])      # enable forced DI
        forced_di_active = True
        writer16(client, P1111, 0, strat['call_kw'])      # all DI low (servo disabled)
        time.sleep(0.05)

        # ---- Mode and command source ----
        writer16(client, P0000, 0, strat['call_kw'])      # Position mode
        writer16(client, P0400, 4, strat['call_kw'])      # Internal multi-segment

        writer32s(client, P0405, UNITS_PER_REV, strat['call_kw'])
        check = reader16(client, P0405, 2, strat['call_kw'])
        print(f"P04.05 now: lo=0x{check[0]:04X}, hi=0x{check[1]:04X}")
        units_per_rev = UNITS_PER_REV

        # ---- Constrain internal program to segment 1 ----
        writer16(client, P1300, 0, strat['call_kw'])      # Single run
        writer16(client, P1301, 1, strat['call_kw'])      # Start at segment 1
        writer16(client, P1302, 1, strat['call_kw'])      # End at segment 1
        writer16(client, P1303, 0, strat['call_kw'])      # Resume unfinished segment
        writer16(client, P1305, 0, strat['call_kw'])      # Incremental position control

        # Drive stores P13.08 as low-word then high-word; keep default ordering.
        writer32s(client, P1308, 0, strat['call_kw'])
        check = reader16(client, P1308, 2, strat['call_kw'])
        print(f"P13.08 reset: hi=0x{check[0]:04X}, lo=0x{check[1]:04X}")

        # ---- Servo ON baseline ----
        base_mask = DI1_MASK
        read_encoder_state(client, strat['call_kw'], units_per_rev)

        # ---- Program segment 1: +180°, dwell ----
        writer32s(client, P1308, deg_to_units(MOVE_DEG, units_per_rev), strat['call_kw'])
        # DIAG:
        check = reader16(client, P1308, 2, strat['call_kw'])
        print(f"P13.08 now: hi=0x{check[0]:04X}, lo=0x{check[1]:04X}")

        writer16(client, P1310, MAX_RPM, strat['call_kw'])
        writer16(client, P1311, ACC_MS, strat['call_kw'])
        writer16(client, P1312, DWELL_MS, strat['call_kw'])

        print("Move 1: +180°")
        set_psec_level(client, strat['call_kw'], base_mask, True)
        time.sleep(2.0)
        set_psec_level(client, strat['call_kw'], base_mask, False)
        read_encoder_state(client, strat['call_kw'], units_per_rev)

        # ---- Program segment 1: -180° (return), no extra dwell ----
        writer32s(client, P1308, -deg_to_units(MOVE_DEG, units_per_rev), strat['call_kw'])
        # DIAG:
        check = reader16(client, P1308, 2, strat['call_kw'])
        print(f"P13.08 now: hi=0x{check[0]:04X}, lo=0x{check[1]:04X}")

        writer16(client, P1312, 0, strat['call_kw'])

        print("Move 2: back to 0°")
        set_psec_level(client, strat['call_kw'], base_mask, True)
        time.sleep(2.0)
        set_psec_level(client, strat['call_kw'], base_mask, False)
        read_encoder_state(client, strat['call_kw'], units_per_rev)

        # Optional feedback
        try:
            spd = reader16(client, P1801, 1, strat['call_kw'])[0]
            print(f"P18.01 speed feedback: {spd}")
        except Exception as e:
            print(f"Note: could not read speed feedback: {e}")

        print("Done.")
    finally:
        call_kw = strat.get('call_kw') if isinstance(strat, dict) else None
        if forced_di_active:
            try:
                writer16(client, P1111, 0, call_kw)
                writer16(client, P1110, 0, call_kw)
            except Exception as e:
                print(f"Note: could not release forced DI: {e}")

        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive the DSY-RS internal multi-segment move.")
    parser.add_argument("--reset", action="store_true",
                        help="Send a P11.01 fault-reset pulse and exit.")
    parser.add_argument("--error", action="store_true",
                        help="Read the current drive error/alarm code (P09.01) and exit.")
    args = parser.parse_args()

    if args.reset or args.error:
        print("pymodbus version:", pymodbus.__version__)
        client = make_client()
        if not client.connect():
            die(f"Failed to open {PORT} at {BAUD} bps")
        try:
            strat = detect_unit_strategy(client, SLAVEID)
            print(f"Per-call kw: {strat['call_kw']}, client attr set: {strat['set_attr']}")
            if args.reset:
                try:
                    addr_echo = reader16(client, P1000, 1, strat['call_kw'])[0]
                    print(f"P10.00 (drive address) reads: {addr_echo}")
                except Exception as e:
                    print(f"Warning: could not read P10.00: {e}")

                print("Ensuring S-ON is OFF before issuing fault reset is recommended.")
                writer16(client, P1110, 1, strat['call_kw'])
                writer16(client, P1111, 0, strat['call_kw'])
                time.sleep(0.05)
                writer16(client, P1101, 1, strat['call_kw'])
                time.sleep(0.05)
                writer16(client, P1101, 0, strat['call_kw'])
                print("Fault reset command sent (P11.01 pulse).")
            if args.error:
                read_error_code(client, strat['call_kw'])
        finally:
            client.close()
    else:
        run_profile()
