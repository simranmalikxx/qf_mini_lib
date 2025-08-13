from datetime import datetime, timedelta

class Date:
    def __init__(self, year: int, month: int, day: int):
        self._dt = datetime(year, month, day)

    # ===== Constructors =====
    @classmethod
    def today(cls):
        now = datetime.now()
        return cls(now.year, now.month, now.day)

    @classmethod
    def from_datetime(cls, dt_obj: datetime):
        return cls(dt_obj.year, dt_obj.month, dt_obj.day)

    # ===== Comparisons =====
    def __lt__(self, other):
        return self._dt < other._dt

    def __le__(self, other):
        return self._dt <= other._dt

    def __eq__(self, other):
        return self._dt == other._dt

    def __gt__(self, other):
        return self._dt > other._dt

    def __ge__(self, other):
        return self._dt >= other._dt

    # ===== Arithmetic =====
    def add_days(self, n: int):
        return Date.from_datetime(self._dt + timedelta(days=n))

    def add_months(self, n: int):
        month = self._dt.month - 1 + n
        year = self._dt.year + month // 12
        month = month % 12 + 1
        day = min(
            self._dt.day,
            [31,
             29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
             31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        )
        return Date(year, month, day)

    def add_years(self, n: int):
        try:
            return Date(self._dt.year + n, self._dt.month, self._dt.day)
        except ValueError:  # handle leap day issue
            return Date(self._dt.year + n, self._dt.month, self._dt.day - 1)

    # ===== Differences =====
    def days_between(self, other) -> int:
        return (other._dt - self._dt).days

    def years_between(self, other) -> float:
        return self.days_between(other) / 365.0

    # ===== Business day helpers =====
    def is_business_day(self) -> bool:
        return self._dt.weekday() < 5  # Monday=0, Sunday=6

    def next_business_day(self):
        next_day = self.add_days(1)
        while not next_day.is_business_day():
            next_day = next_day.add_days(1)
        return next_day

    # ===== Representation =====
    def __str__(self):
        return self._dt.strftime("%Y-%m-%d")

    def __repr__(self):
        return f"Date({self._dt.year}, {self._dt.month}, {self._dt.day})"

    # ===== Internal accessor =====
    @property
    def datetime(self):
        return self._dt
