import math

def fmt(n):
    millnames = ['', 'Thousand', 'Million', 'Billion', 'Trillion']

    n = float(n)
    millidx = max(0, min(len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.0f} {}'.format(n / 10 ** (3 * millidx), millnames[millidx])

if __name__ == '__main__':
    for n in (1.23456789 * 10 ** r for r in range(-2, 19, 1)):
	print '{} : {}'.format(n, fmt(n))