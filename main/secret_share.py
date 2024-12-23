import random
import sys


mod = (2**255) - 19

def fastpow(x, pow, mod):
    ret = 1
    while(pow):
        if((pow % 2) == 1):
            ret = ret * x % mod
        x = x * x % mod
        pow = pow // 2
    return ret

def inv(x, mod):
    return fastpow(x, mod - 2, mod)

class prime_field:
    def __init__(self, value = 0):
        self.val = value
    def __add__(self, b):
        ret = prime_field()
        ret.val = self.val + b.val
        ret.val = ret.val % mod
        return ret
    def __mul__(self, b):
        ret = prime_field()
        ret.val = self.val * b.val
        ret.val = ret.val % mod
        return ret
    def __sub__(self, b):
        ret = prime_field()
        ret.val = self.val - b.val
        if(ret.val < 0):
            ret.val = ret.val + mod
        assert(ret.val >= 0 and ret.val < mod)
        return ret
    def __truediv__(self, b):
        ret = prime_field()
        ret.val = self.val * inv(b.val, mod) % mod
        assert(ret.val * b.val % mod == self.val)
        return ret
    #def fill_high_bits(self):
        #self.val |= random.randint(2**128, (2**255) - 19)
    def to_bytes(self):
        return self.val.to_bytes(256 // 8, 'little')
    def from_bytes(self, data):
        self.val.from_bytes(data, 'little')

class polynomial:
    def __init__(self, deg, secret):
        self.coef = []
        self.coef.append(secret)
        self.deg = deg
        for i in range(1, deg):
            self.coef.append(prime_field(random.randint(0, (2**255) - 19)))
    def eval(self, x):
        res = prime_field()
        x_n = prime_field(1)
        x = prime_field(x)
        for i in range(self.deg):
            res = res + x_n * self.coef[i]
            x_n = x_n * x
        return res
    def reconstruct(self, x, y):
        pass

def construct(bin_data, out_files, n, t):
    bin_data = bytearray(bin_data)
    length = len(bin_data)
    
    sec_pack = 128 // 8
    
    for i in range((sec_pack - (length % sec_pack)) % sec_pack):
        bin_data.append(0)
    padded_len = len(bin_data)
    print(length, padded_len)
    assert(padded_len % sec_pack == 0)

    out_bytes = []
    for i in range(n):
        out_bytes.append(bytearray())
        out_bytes[i] += length.to_bytes(8, 'little')
        out_bytes[i] += padded_len.to_bytes(8, 'little')
        out_bytes[i] += (i + 1).to_bytes(8, 'little')
        out_bytes[i] += t.to_bytes(8, 'little')


    for i in range(padded_len // sec_pack):
        sec = prime_field()
        #sec.fill_high_bits()
        sec.val |= int.from_bytes(bin_data[i * sec_pack : i * sec_pack + sec_pack], 'little')
        poly = polynomial(t, sec)
        for j in range(n):
            res = poly.eval(j + 1)
            out_bytes[j] += res.to_bytes()
            print(res.val)
        print('secret = ' + str(poly.eval(0).val))
        print(poly.eval(0).val.to_bytes(sec_pack, 'little'))
        print()
    for i in range(n):
        out_files[i].write(out_bytes[i])

def lagrange_recons(x_s, y_s, t):
    res = prime_field(0)
    x = prime_field(0)
    for i in range(t):
        y = y_s[i]
        Lx = prime_field(1)
        for j in range(t):
            if i == j:
                continue
            Lx = Lx * ((x - x_s[j]) / (x_s[i] - x_s[j]))
        res = res + Lx * y
    return res

def reconstruct(in_files, out_file, n):
    in_bytes = [x.read() for x in in_files]
    length = int.from_bytes(in_bytes[0][0 : 8], 'little')
    padded_len = int.from_bytes(in_bytes[0][8 : 16], 'little')
    t = int.from_bytes(in_bytes[0][24 : 32], 'little')
    sec_pack = 128 // 8
    coef_pack = 256 // 8
    if n < t:
        print('Not sufficient files to reconstruct', n, t)
        return
    else:
        x_s = []
        for i in range(n):
            val = int.from_bytes(in_bytes[i][16:24], 'little')
            x_s.append(prime_field(val))
        out_bytes = bytearray()
        for i in range(padded_len // sec_pack):
            y_s = []
            for j in range(n):
                val = int.from_bytes(in_bytes[j][32 + i * coef_pack: 32 + (i + 1) * coef_pack], 'little')
                y_s.append(prime_field(val))
                print(val)
            secret = lagrange_recons(x_s, y_s, t)
            print(secret.val)
            out_bytes += secret.val.to_bytes(sec_pack, 'little')
        
        out_bytes = out_bytes[:length]
        print(out_bytes)
        out_file.write(out_bytes)

if(len(sys.argv[1]) < 2):

    print("Usage: python secret_share.py MODE param1 param2 ... paramk")
    print("For MODE = 0, python secret_share.py 0 N T output_file1 output_file2 ... output_fileN")
    print("MODE = 0 will generate secret share files, N is the number of shares, T is the threshold.")
    print("For MODE = 1, python secret_share.py 1 T plaintext_file input_file1 input_file2 ... input_fileT")
    print("MODE = 1 will reconstruct the plaintext, T is the threshold")

mode = int(sys.argv[1])

if mode == 0: #construct mode

    path = sys.argv[2]
    n = int(sys.argv[3])
    t = int(sys.argv[4])
    outfile_name = sys.argv[5]


    in_file = open(path, 'rb')
    out_files = [open(outfile_name + str(i), 'wb') for i in range(n)]

    bin_data = in_file.read()
    print(bin_data)

    construct(bin_data, out_files, n, t)

    in_file.close()
    for i in range(n):
        out_files[i].close()
else: #reconstruct mode
    n = int(sys.argv[2])
    outfile_name = sys.argv[3]
    in_files = [open(sys.argv[4 + i], 'rb') for i in range(n)]
    out_file = open(outfile_name, 'wb')

    reconstruct(in_files, out_file, n)

    for f in in_files:
        f.close()
    out_file.close()