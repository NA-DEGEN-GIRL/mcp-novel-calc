#!/usr/bin/env python3
"""
소설 집필용 계산기 MCP Server
- 수학 수식, 날짜 계산, D+ 변환, 속도/거리/시간, 단위 변환
"""

import ast
import math
import operator
import calendar
from datetime import datetime, timedelta, date
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("novel-calc")

WEEKDAY_KR = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

# =========================================
# 1. 안전한 수식 계산
# =========================================

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(node):
    """AST 기반 안전한 수식 평가"""
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"허용되지 않는 상수: {node.value}")
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"허용되지 않는 연산자: {type(node.op).__name__}")
        return op(safe_eval(node.left), safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"허용되지 않는 단항 연산자: {type(node.op).__name__}")
        return op(safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            func = SAFE_FUNCTIONS[node.func.id]
            if callable(func):
                args = [safe_eval(arg) for arg in node.args]
                return func(*args)
            return func  # pi, e 등 상수
        raise ValueError(f"허용되지 않는 함수: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            val = SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val  # pi, e
        raise ValueError(f"허용되지 않는 이름: {node.id}")
    elif isinstance(node, ast.Tuple):
        return tuple(safe_eval(el) for el in node.elts)
    elif isinstance(node, ast.List):
        return [safe_eval(el) for el in node.elts]
    else:
        raise ValueError(f"허용되지 않는 구문: {type(node).__name__}")


@mcp.tool()
def calculate(expression: str) -> str:
    """수학 수식을 계산합니다.

    사칙연산, 거듭제곱(**), 제곱근(sqrt), 삼각함수, 로그, 반올림 등을 지원합니다.

    Args:
        expression: 수학 수식. 예: "1250 * 1.35", "sqrt(144)", "round(3.14159, 2)", "2**10"
    """
    try:
        import re
        # 숫자 내 천 단위 쉼표만 제거 (1,250,000 → 1250000), 함수 인자 쉼표는 유지
        expr = re.sub(r'(\d),(\d{3})', r'\1\2', expression)
        expr = re.sub(r'(\d),(\d{3})', r'\1\2', expr)  # 반복 (백만 이상)
        tree = ast.parse(expr, mode="eval")
        result = safe_eval(tree)
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result = int(result)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


# =========================================
# 2. 날짜 계산
# =========================================

def parse_date(s: str) -> date:
    """다양한 형식의 날짜 문자열을 파싱"""
    s = s.strip().replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"날짜 형식을 인식할 수 없습니다: {s} (YYYY-MM-DD 형식을 사용하세요)")


def format_date(d: date) -> str:
    """날짜를 한국어 형식으로 포맷"""
    wd = WEEKDAY_KR[d.weekday()]
    return f"{d.year}년 {d.month}월 {d.day}일 ({wd})"


def add_months(d: date, months: int) -> date:
    """월 단위 더하기/빼기"""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@mcp.tool()
def date_calc(
    date_str: str,
    operation: str = "add",
    days: int = 0,
    months: int = 0,
    years: int = 0,
) -> str:
    """날짜에 일/월/년을 더하거나 뺍니다.

    Args:
        date_str: 기준 날짜 (YYYY-MM-DD). 예: "2020-03-12"
        operation: "add" 또는 "subtract"
        days: 더할/뺄 일수
        months: 더할/뺄 월수
        years: 더할/뺄 년수
    """
    try:
        d = parse_date(date_str)
        sign = 1 if operation == "add" else -1
        result = add_months(d, sign * (months + years * 12))
        result = result + timedelta(days=sign * days)

        parts = []
        if years: parts.append(f"{years}년")
        if months: parts.append(f"{months}월")
        if days: parts.append(f"{days}일")
        desc = " ".join(parts) if parts else "0일"
        op_str = "+" if operation == "add" else "-"

        return f"{format_date(d)} {op_str} {desc} = {format_date(result)} ({result.isoformat()})"
    except Exception as e:
        return f"오류: {e}"


@mcp.tool()
def date_diff(date1: str, date2: str) -> str:
    """두 날짜 사이의 차이를 계산합니다.

    Args:
        date1: 첫 번째 날짜 (YYYY-MM-DD)
        date2: 두 번째 날짜 (YYYY-MM-DD)
    """
    try:
        d1 = parse_date(date1)
        d2 = parse_date(date2)
        delta = abs((d2 - d1).days)
        weeks = delta / 7
        months_approx = delta / 30.44
        years_approx = delta / 365.25

        lines = [
            f"{format_date(d1)} ~ {format_date(d2)}",
            f"",
            f"차이: {delta}일",
            f"  = 약 {weeks:.1f}주",
            f"  = 약 {months_approx:.1f}개월",
        ]
        if years_approx >= 1:
            lines.append(f"  = 약 {years_approx:.1f}년")

        return "\n".join(lines)
    except Exception as e:
        return f"오류: {e}"


# =========================================
# 3. D+ 변환
# =========================================

@mcp.tool()
def d_plus(d_day: str, target: str) -> str:
    """D-Day 기준으로 D+N 날짜를 구하거나, 특정 날짜가 D+며칠인지 계산합니다.

    Args:
        d_day: D-Day 기준일 (YYYY-MM-DD). 예: "2020-03-12"
        target: D+ 숫자 (예: "100") 또는 날짜 (예: "2021-01-01")
    """
    try:
        base = parse_date(d_day)

        # target이 숫자인지 확인
        try:
            n = int(target)
            result = base + timedelta(days=n)
            return f"D+0 = {format_date(base)}\nD+{n} = {format_date(result)} ({result.isoformat()})"
        except ValueError:
            pass

        # target이 날짜인 경우
        target_date = parse_date(target)
        delta = (target_date - base).days
        sign = "+" if delta >= 0 else ""
        return f"D+0 = {format_date(base)}\n{format_date(target_date)} = D{sign}{delta}"

    except Exception as e:
        return f"오류: {e}"


# =========================================
# 4. 요일 조회
# =========================================

@mcp.tool()
def weekday(date_str: str) -> str:
    """특정 날짜의 요일과 주차 정보를 반환합니다.

    Args:
        date_str: 날짜 (YYYY-MM-DD). 예: "2020-03-12"
    """
    try:
        d = parse_date(date_str)
        wd = WEEKDAY_KR[d.weekday()]
        iso_year, iso_week, _ = d.isocalendar()
        month_week = (d.day - 1) // 7 + 1

        lines = [
            f"{d.isoformat()} = {wd}",
            f"",
            f"ISO 주차: {iso_year}년 제{iso_week}주",
            f"월 내 주차: {d.month}월 {month_week}째주",
        ]

        # 해당 월 달력
        lines.append(f"")
        lines.append(f"── {d.year}년 {d.month}월 ──")
        lines.append("월  화  수  목  금  토  일")
        cal = calendar.monthcalendar(d.year, d.month)
        for week in cal:
            row = ""
            for day_num in week:
                if day_num == 0:
                    row += "    "
                elif day_num == d.day:
                    row += f"[{day_num:2d}]"
                else:
                    row += f" {day_num:2d} "
            lines.append(row)

        return "\n".join(lines)
    except Exception as e:
        return f"오류: {e}"


# =========================================
# 5. 속도·거리·시간
# =========================================

@mcp.tool()
def speed_distance_time(
    speed: float = 0,
    distance: float = 0,
    time: float = 0,
    unit: str = "kmh",
) -> str:
    """속도, 거리, 시간 중 2개를 입력하면 나머지를 계산합니다.

    Args:
        speed: 속도 (km/h 또는 m/s, 0이면 계산 대상)
        distance: 거리 (km 또는 m, 0이면 계산 대상)
        time: 시간 (시간 또는 초, 0이면 계산 대상)
        unit: "kmh" (km/h, km, 시간) 또는 "ms" (m/s, m, 초)
    """
    speed_unit = "km/h" if unit == "kmh" else "m/s"
    dist_unit = "km" if unit == "kmh" else "m"
    time_unit = "시간" if unit == "kmh" else "초"

    given = sum(1 for v in [speed, distance, time] if v > 0)
    if given != 2:
        return "오류: 3개 값 중 정확히 2개를 입력하세요 (계산할 값은 0으로)."

    try:
        if speed == 0:
            speed = distance / time
            result_label = "속도"
            result_value = f"{speed:.2f} {speed_unit}"
        elif distance == 0:
            distance = speed * time
            result_label = "거리"
            result_value = f"{distance:.2f} {dist_unit}"
        else:
            time = distance / speed
            result_label = "시간"
            if unit == "kmh" and time < 1:
                result_value = f"{time * 60:.1f}분"
            elif unit == "kmh" and time >= 24:
                days = time / 24
                result_value = f"{time:.2f}{time_unit} (약 {days:.1f}일)"
            else:
                result_value = f"{time:.2f} {time_unit}"

        lines = [
            f"주어진 값:",
        ]
        if speed > 0 and result_label != "속도":
            lines.append(f"  속도: {speed} {speed_unit}")
        if distance > 0 and result_label != "거리":
            lines.append(f"  거리: {distance} {dist_unit}")
        if time > 0 and result_label != "시간":
            lines.append(f"  시간: {time} {time_unit}")

        lines.append(f"")
        lines.append(f"결과: {result_label} = {result_value}")

        return "\n".join(lines)
    except ZeroDivisionError:
        return "오류: 0으로 나눌 수 없습니다."


# =========================================
# 6. 단위 변환
# =========================================

# 모든 단위를 기준 단위로 변환하는 계수
# 거리: 기준 = m
DISTANCE_UNITS = {
    "m": 1.0,
    "미터": 1.0,
    "km": 1000.0,
    "킬로미터": 1000.0,
    "cm": 0.01,
    "센티미터": 0.01,
    "리": 392.727,      # 1리 ≈ 392.727m (조선 시대 기준)
    "里": 392.727,
    "장": 3.03,         # 1장(丈) ≈ 3.03m
    "丈": 3.03,
    "척": 0.303,        # 1척(尺) ≈ 30.3cm
    "尺": 0.303,
    "촌": 0.0303,       # 1촌(寸) ≈ 3.03cm
    "寸": 0.0303,
    "보": 1.2,          # 1보(步) ≈ 1.2m (걸음)
    "步": 1.2,
}

# 시간: 기준 = 분
TIME_UNITS = {
    "분": 1.0,
    "초": 1 / 60,
    "시간": 60.0,
    "일": 1440.0,
    "시진": 120.0,     # 1시진(時辰) = 2시간
    "時辰": 120.0,
    "각": 15.0,        # 1각(刻) = 15분
    "刻": 15.0,
    "경": 120.0,       # 1경(更) = 약 2시간 (야간 5경)
}

# 무게: 기준 = g
WEIGHT_UNITS = {
    "g": 1.0,
    "그램": 1.0,
    "kg": 1000.0,
    "킬로그램": 1000.0,
    "근": 600.0,       # 1근(斤) = 600g
    "斤": 600.0,
    "냥": 37.5,        # 1냥(兩) = 37.5g
    "兩": 37.5,
    "돈": 3.75,        # 1돈 = 3.75g
    "관": 3750.0,      # 1관(貫) = 3.75kg
}

UNIT_CATEGORIES = {
    "거리": DISTANCE_UNITS,
    "시간": TIME_UNITS,
    "무게": WEIGHT_UNITS,
}


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """단위를 변환합니다. 거리(km, m, 리, 장, 척), 시간(시간, 분, 시진, 각), 무게(kg, 근, 냥) 등을 지원합니다.

    Args:
        value: 변환할 값
        from_unit: 원래 단위. 예: "리", "km", "시진", "근"
        to_unit: 변환할 단위. 예: "km", "m", "시간", "kg"
    """
    for cat_name, units in UNIT_CATEGORIES.items():
        if from_unit in units and to_unit in units:
            base_value = value * units[from_unit]
            result = base_value / units[to_unit]
            if result == int(result) and abs(result) < 1e15:
                result = int(result)
            else:
                result = round(result, 4)
            line = f"{value} {from_unit} = {result} {to_unit} ({cat_name})"
            # 거리인 경우 서술형 참고 추가
            if cat_name == "거리" and to_unit in ("km", "킬로미터", "m", "미터"):
                meters = base_value
                if meters <= 500:
                    line += f"\n📝 약 {meters:.0f}m — 활 사거리 정도"
                elif meters <= 2000:
                    line += f"\n📝 약 {meters/1000:.1f}km — 성인 걸음으로 {meters/1000/4*60:.0f}분 거리"
                else:
                    hours = meters / 1000 / 4
                    line += f"\n📝 약 {meters/1000:.1f}km — 보행 {hours:.1f}시간, 말(보통) {meters/1000/15:.1f}시간 거리"
            return line

    # 단위를 찾지 못한 경우 가능한 단위 목록 표시
    all_units = {}
    for cat_name, units in UNIT_CATEGORIES.items():
        all_units[cat_name] = sorted(set(units.keys()))

    lines = [f"오류: '{from_unit}' → '{to_unit}' 변환을 지원하지 않습니다.", ""]
    lines.append("지원 단위:")
    for cat_name, units in all_units.items():
        lines.append(f"  {cat_name}: {', '.join(units)}")
    return "\n".join(lines)


# =========================================
# 7. 십이지시(十二支時) 변환
# =========================================

TWELVE_BRANCHES = [
    ("자시", "子時", 23, 1, "쥐"),
    ("축시", "丑時", 1, 3, "소"),
    ("인시", "寅時", 3, 5, "범"),
    ("묘시", "卯時", 5, 7, "토끼"),
    ("진시", "辰時", 7, 9, "용"),
    ("사시", "巳時", 9, 11, "뱀"),
    ("오시", "午時", 11, 13, "말"),
    ("미시", "未時", 13, 15, "양"),
    ("신시", "申時", 15, 17, "원숭이"),
    ("유시", "酉時", 17, 19, "닭"),
    ("술시", "戌時", 19, 21, "개"),
    ("해시", "亥時", 21, 23, "돼지"),
]


@mcp.tool()
def convert_time(time_str: str) -> str:
    """현대 시각을 십이지시(동양식 시간)로 변환하거나, 십이지시를 현대 시각으로 변환합니다.

    Args:
        time_str: 현대 시각 (예: "14:30", "02:00") 또는 십이지시 (예: "미시", "자시", "子時")
    """
    # 십이지시 → 현대 시각
    for kr, cn, start, end, animal in TWELVE_BRANCHES:
        if time_str.strip() in (kr, cn):
            if start > end:  # 자시 (23~01)
                return (
                    f"{kr}({cn}) = {start:02d}:00 ~ 익일 {end:02d}:00 (약 2시간)\n"
                    f"초각(初刻) = {start:02d}:00, 이각 = {start:02d}:30, 삼각 = {(start+1)%24:02d}:00, 사각 = {(start+1)%24:02d}:30\n"
                    f"십이지: {animal}"
                )
            return (
                f"{kr}({cn}) = {start:02d}:00 ~ {end:02d}:00 (약 2시간)\n"
                f"초각(初刻) = {start:02d}:00, 이각 = {start:02d}:30, 삼각 = {start+1:02d}:00, 사각 = {start+1:02d}:30\n"
                f"십이지: {animal}"
            )

    # 현대 시각 → 십이지시
    try:
        parts = time_str.strip().replace("시", ":").replace("분", "").split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0

        for kr, cn, start, end, animal in TWELVE_BRANCHES:
            if start > end:  # 자시
                if hour >= start or hour < end:
                    gak = _calc_gak(hour, minute, start)
                    return f"{hour:02d}:{minute:02d} = {kr}({cn}) {gak}\n범위: {start:02d}:00 ~ 익일 {end:02d}:00"
            else:
                if start <= hour < end:
                    gak = _calc_gak(hour, minute, start)
                    return f"{hour:02d}:{minute:02d} = {kr}({cn}) {gak}\n범위: {start:02d}:00 ~ {end:02d}:00"

        return "오류: 시각을 판별할 수 없습니다."
    except (ValueError, IndexError):
        return (
            "오류: 형식을 인식할 수 없습니다.\n"
            "현대 시각: '14:30', '02:00' 또는 십이지시: '미시', '자시', '子時'"
        )


def _calc_gak(hour, minute, branch_start):
    """시진 내에서 몇 각(刻)인지 계산"""
    offset = ((hour - branch_start) % 24) * 60 + minute
    gak_names = ["초각(初刻)", "이각(二刻)", "삼각(三刻)", "사각(四刻)"]
    idx = min(offset // 30, 3)
    return gak_names[idx]


# =========================================
# 8. 화폐 계산 (무협용)
# =========================================

CURRENCY_SYSTEMS = {
    "동양_기본": {
        "name": "동양(무협) 기본 화폐",
        "units": [
            ("금", 10000),    # 1금 = 10000문
            ("냥", 1000),     # 1냥(은자) = 1000문
            ("전", 100),      # 1전 = 100문
            ("푼", 1),        # 1푼(문) = 기본 단위
        ],
        "base": "푼",
        "aliases": {"문": "푼", "文": "푼", "錢": "전", "兩": "냥"},
    },
}


@mcp.tool()
def currency_calc(amount: float, unit: str, system: str = "동양_기본") -> str:
    """화폐 단위를 변환합니다 (무협 소설용).

    1금 = 10냥, 1냥(은자) = 10전, 1전 = 100푼(문).

    Args:
        amount: 금액
        unit: 화폐 단위. 예: "냥", "전", "푼", "문", "금"
        system: 화폐 체계 (기본: "동양_기본")
    """
    sys = CURRENCY_SYSTEMS.get(system)
    if not sys:
        return f"오류: '{system}' 화폐 체계를 찾을 수 없습니다."

    # 별칭 처리
    unit = sys["aliases"].get(unit, unit)

    # 기본 단위(푼)로 변환
    base_value = None
    for u_name, u_rate in sys["units"]:
        if u_name == unit:
            base_value = amount * u_rate
            break
    if base_value is None:
        units_list = ", ".join(u[0] for u in sys["units"])
        return f"오류: '{unit}' 단위를 찾을 수 없습니다. 사용 가능: {units_list}"

    base_value = int(base_value)

    # 각 단위로 분해
    lines = [f"{amount:g} {unit} = {base_value:,} {sys['base']}(문)", ""]
    remainder = base_value
    breakdown = []
    for u_name, u_rate in sys["units"]:
        if remainder >= u_rate:
            count = remainder // u_rate
            remainder %= u_rate
            breakdown.append(f"{count:g}{u_name}")
    if breakdown:
        lines.append(f"분해: {' '.join(breakdown)}")

    # 물가 참고 (대략적)
    lines.append("")
    lines.append("── 참고 물가 (무협 일반 설정) ──")
    lines.append(f"  국수 한 그릇: 5~10문")
    lines.append(f"  객잔 1박: 50~100문")
    lines.append(f"  좋은 식사: 20~50문")
    lines.append(f"  평범한 검: 2~5냥")
    lines.append(f"  좋은 말 한 필: 10~30냥")

    return "\n".join(lines)


# =========================================
# 9. 부피(곡물) 단위
# =========================================

VOLUME_UNITS = {
    "홉": 1.0,       # 기준 = 홉 (약 180ml)
    "合": 1.0,
    "되": 10.0,      # 1되 = 10홉
    "升": 10.0,
    "말": 100.0,     # 1말 = 10되
    "斗": 100.0,
    "섬": 1000.0,    # 1섬 = 10말
    "石": 1000.0,
}

UNIT_CATEGORIES["부피(곡물)"] = VOLUME_UNITS


@mcp.tool()
def supply_calc(
    people: int,
    consumption_per_day: float,
    consumption_unit: str = "홉",
    supply_amount: float = 0,
    supply_unit: str = "섬",
    days: int = 0,
) -> str:
    """군량/보급품 계산. 인원수와 1인당 소비량으로 필요 보급량 또는 버틸 수 있는 일수를 계산합니다.

    Args:
        people: 인원수
        consumption_per_day: 1인 1일 소비량 (기본 단위: 홉)
        consumption_unit: 소비량 단위 (홉/되/말/섬)
        supply_amount: 보유 보급량 (0이면 일수 기반으로 필요량 계산)
        supply_unit: 보급량 단위 (홉/되/말/섬)
        days: 목표 일수 (0이면 보유량 기반으로 버틸 일수 계산)
    """
    # 모든 값을 홉으로 변환
    cons_per_day_hop = consumption_per_day * VOLUME_UNITS.get(consumption_unit, 1)
    daily_total_hop = people * cons_per_day_hop

    lines = [
        f"인원: {people:,}명",
        f"1인 1일 소비: {consumption_per_day:g} {consumption_unit} ({cons_per_day_hop:g}홉)",
        f"전체 1일 소비: {daily_total_hop:,.0f}홉 = {daily_total_hop/100:.1f}말 = {daily_total_hop/1000:.2f}섬",
        "",
    ]

    if supply_amount > 0 and days == 0:
        # 보유량으로 버틸 일수 계산
        supply_hop = supply_amount * VOLUME_UNITS.get(supply_unit, 1)
        can_days = supply_hop / daily_total_hop if daily_total_hop > 0 else 0
        lines.append(f"보유량: {supply_amount:g} {supply_unit} ({supply_hop:,.0f}홉)")
        lines.append(f"버틸 수 있는 일수: {can_days:.1f}일 (약 {can_days/30:.1f}개월)")
    elif days > 0:
        # 필요 보급량 계산
        need_hop = daily_total_hop * days
        lines.append(f"목표 기간: {days}일")
        lines.append(f"필요 보급량: {need_hop:,.0f}홉 = {need_hop/100:,.1f}말 = {need_hop/1000:,.1f}섬")
    else:
        lines.append("supply_amount 또는 days 중 하나를 입력하세요.")

    return "\n".join(lines)


# =========================================
# 10. 이동 수단별 거리/시간 예측
# =========================================

TRAVEL_PRESETS = {
    "보행": {"speed": 4.0, "desc": "성인 보통 걸음"},
    "급행군": {"speed": 6.0, "desc": "군대 급행군"},
    "말_보통": {"speed": 15.0, "desc": "말(보통 속도)"},
    "말_질주": {"speed": 40.0, "desc": "말(전력 질주, 단시간)"},
    "천리마": {"speed": 45.0, "desc": "천리마(질주)"},
    "마차": {"speed": 8.0, "desc": "마차(평지)"},
    "경공_초급": {"speed": 30.0, "desc": "경공(초급)"},
    "경공_중급": {"speed": 50.0, "desc": "경공(중급)"},
    "경공_고수": {"speed": 80.0, "desc": "경공(고수/절정)"},
    "경공_초절정": {"speed": 120.0, "desc": "경공(초절정/화경)"},
    "배": {"speed": 10.0, "desc": "범선(순풍)"},
}


@mcp.tool()
def travel_estimate(distance: float, mode: str, rest_hours: float = 0) -> str:
    """이동 수단별 소요 시간을 추정합니다.

    Args:
        distance: 이동 거리 (km). 리 단위면 먼저 unit_convert로 변환하세요.
        mode: 이동 수단. 보행, 급행군, 말_보통, 말_질주, 천리마, 마차, 경공_초급, 경공_중급, 경공_고수, 경공_초절정, 배
        rest_hours: 중간 휴식 시간 (시간 단위, 기본 0)
    """
    preset = TRAVEL_PRESETS.get(mode)
    if not preset:
        modes = "\n".join(f"  {k}: {v['desc']} ({v['speed']}km/h)" for k, v in TRAVEL_PRESETS.items())
        return f"오류: '{mode}'를 찾을 수 없습니다.\n\n사용 가능 수단:\n{modes}"

    speed = preset["speed"]
    travel_hours = distance / speed
    total_hours = travel_hours + rest_hours

    # 시간 포맷
    def fmt_time(h):
        if h < 1:
            return f"{h*60:.0f}분"
        days = h / 24
        if days >= 1:
            return f"{h:.1f}시간 (약 {days:.1f}일)"
        return f"{h:.1f}시간"

    # 리 단위도 함께 표시
    distance_ri = distance / 0.392727

    lines = [
        f"이동 수단: {preset['desc']} ({speed}km/h)",
        f"이동 거리: {distance:.1f}km (약 {distance_ri:.0f}리)",
        f"",
        f"순수 이동: {fmt_time(travel_hours)}",
    ]
    if rest_hours > 0:
        lines.append(f"휴식 시간: {rest_hours}시간")
        lines.append(f"총 소요: {fmt_time(total_hours)}")

    # 시진 표현
    travel_sijin = travel_hours / 2
    lines.append(f"")
    lines.append(f"동양식: 약 {travel_sijin:.1f}시진(時辰)")

    # 서술형 팁
    lines.append(f"")
    if distance_ri <= 10:
        lines.append(f"📝 \"{distance_ri:.0f}리 거리는 {preset['desc']}으로 {fmt_time(travel_hours)}이면 닿을 수 있다.\"")
    else:
        lines.append(f"📝 \"{distance_ri:.0f}리 길을 {preset['desc']}으로 달리면 {fmt_time(travel_hours)} 정도 걸린다.\"")

    return "\n".join(lines)


# =========================================
# 11. 복리/성장 계산 (LitRPG/수련용)
# =========================================

@mcp.tool()
def growth_calc(
    start_value: float,
    target_value: float = 0,
    daily_rate: float = 0,
    days: int = 0,
) -> str:
    """복리(지수) 성장을 계산합니다. 내공, 경험치, 팔로워 등의 성장 시뮬레이션에 유용합니다.

    3개 값 중 1개를 0으로 두면 나머지 2개로 계산합니다.

    Args:
        start_value: 시작 값 (예: 현재 내공 100)
        target_value: 목표 값 (예: 목표 내공 300, 0이면 계산)
        daily_rate: 일일 성장률 (%, 예: 1.5 → 매일 1.5% 성장, 0이면 계산)
        days: 기간 (일, 0이면 계산)
    """
    import math

    # target_value 계산
    if target_value == 0 and daily_rate > 0 and days > 0:
        rate = 1 + daily_rate / 100
        result = start_value * (rate ** days)
        return (
            f"시작: {start_value:,.1f}\n"
            f"성장률: 일 {daily_rate}% (복리)\n"
            f"기간: {days}일\n\n"
            f"결과: {result:,.1f} ({result/start_value:.1f}배)\n\n"
            f"📝 \"{days}일간 매일 {daily_rate}%씩 성장하면 {start_value:,.0f}에서 {result:,.0f}이 된다.\""
        )

    # days 계산
    if days == 0 and daily_rate > 0 and target_value > 0:
        rate = 1 + daily_rate / 100
        if rate <= 1:
            return "오류: 성장률이 0 이하입니다."
        needed = math.log(target_value / start_value) / math.log(rate)
        needed = math.ceil(needed)
        return (
            f"시작: {start_value:,.1f} → 목표: {target_value:,.1f} ({target_value/start_value:.1f}배)\n"
            f"성장률: 일 {daily_rate}%\n\n"
            f"필요 기간: {needed}일 (약 {needed/30:.1f}개월)\n\n"
            f"📝 \"매일 {daily_rate}%씩 수련하면 {needed}일이면 {target_value:,.0f}에 도달한다.\""
        )

    # daily_rate 계산
    if daily_rate == 0 and days > 0 and target_value > 0:
        rate = (target_value / start_value) ** (1 / days)
        pct = (rate - 1) * 100
        return (
            f"시작: {start_value:,.1f} → 목표: {target_value:,.1f}\n"
            f"기간: {days}일\n\n"
            f"필요 일일 성장률: {pct:.3f}%\n\n"
            f"📝 \"{days}일 만에 {target_value/start_value:.1f}배가 되려면 매일 {pct:.2f}%씩 성장해야 한다.\""
        )

    return "오류: target_value, daily_rate, days 중 하나를 0으로 두면 나머지로 계산합니다."


# =========================================
# 서버 시작
# =========================================

if __name__ == "__main__":
    mcp.run()
