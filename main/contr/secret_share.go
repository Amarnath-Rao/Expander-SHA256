package secret_sharing

import (
	"crypto/rand"
	"io"
	"math/big"

	"github.com/ethereum/go-ethereum/crypto/secp256k1"
)

type CurveElement struct {
	x, y *big.Int
}

var curve = secp256k1.S256()

func (fe *CurveElement) Add(b *CurveElement) *CurveElement {
	newx, newy := curve.Add(fe.x, fe.y, b.x, b.y)
	return &CurveElement{x: newx, y: newy}
}

func (fe *CurveElement) ScalarMul(b *big.Int) *CurveElement {
	newx, newy := curve.ScalarBaseMult(b.Bytes())
	return &CurveElement{x: newx, y: newy}
}

var mod = curve.Params().N

func fastpow(x, pow, mod *big.Int) *big.Int {
	ret := big.NewInt(1)
	for pow.Sign() > 0 {
		if pow.Bit(0) == 1 {
			ret.Mul(ret, x)
			ret.Mod(ret, mod)
		}
		x.Mul(x, x)
		x.Mod(x, mod)
		pow.Rsh(pow, 1)
	}
	return ret
}

func inv(x, mod *big.Int) *big.Int {
	return fastpow(x, new(big.Int).Sub(mod, big.NewInt(2)), mod)
}

type PrimeField struct {
	val *big.Int
}

func NewPrimeField(value int64) *PrimeField {
	return &PrimeField{val: big.NewInt(value)}
}

func (pf *PrimeField) Add(b *PrimeField) *PrimeField {
	ret := &PrimeField{val: new(big.Int)}
	ret.val.Add(pf.val, b.val)
	ret.val.Mod(ret.val, mod)
	return ret
}

func (pf *PrimeField) Mul(b *PrimeField) *PrimeField {
	ret := &PrimeField{val: new(big.Int)}
	ret.val.Mul(pf.val, b.val)
	ret.val.Mod(ret.val, mod)
	return ret
}

func (pf *PrimeField) Sub(b *PrimeField) *PrimeField {
	ret := &PrimeField{val: new(big.Int)}
	ret.val.Sub(pf.val, b.val)
	if ret.val.Sign() < 0 {
		ret.val.Add(ret.val, mod)
	}
	return ret
}

func (pf *PrimeField) Div(b *PrimeField) *PrimeField {
	ret := &PrimeField{val: new(big.Int)}
	inv := inv(b.val, mod)
	ret.val.Mul(pf.val, inv)
	ret.val.Mod(ret.val, mod)
	return ret
}

func (pf *PrimeField) ToBytes() []byte {
	return pf.val.Bytes()
}

func (pf *PrimeField) FromBytes(data []byte) {
	pf.val.SetBytes(data)
}

type Polynomial struct {
	coef []*PrimeField
	deg  int
}

func NewPolynomial(deg int, secret *PrimeField) *Polynomial {
	p := &Polynomial{
		coef: make([]*PrimeField, deg),
		deg:  deg,
	}
	p.coef[0] = secret
	for i := 1; i < deg; i++ {
		randVal, _ := rand.Int(rand.Reader, mod)
		p.coef[i] = &PrimeField{val: randVal}
	}
	return p
}

func (p *Polynomial) Eval(x int64) *PrimeField {
	res := NewPrimeField(0)
	xn := NewPrimeField(1)
	xPF := NewPrimeField(x)
	for i := 0; i < p.deg; i++ {
		res = res.Add(xn.Mul(p.coef[i]))
		xn = xn.Mul(xPF)
	}
	return res
}

func Construct(dA *big.Int, outFilesValues []io.Writer, outFilesXs []io.Writer, n, t int) { // dA is the private key
	dAPF := &PrimeField{val: new(big.Int).SetBytes(dA.Bytes())}
	polynomial := NewPolynomial(t, dAPF) // construct the polynomial f, where f(0) = dA, and the share is f(i), i = 1, 2, ..., n
	for i := 0; i < n; i++ {
		share := polynomial.Eval(int64(i + 1))
		outFilesValues[i].Write(share.ToBytes())
		outFilesXs[i].Write(big.NewInt(int64(i + 1)).Bytes())
	}
}

func lagrangeRecons(xs, ys []*PrimeField, t int) *PrimeField {
	res := NewPrimeField(0)
	x := NewPrimeField(0)
	for i := 0; i < t; i++ {
		y := ys[i]
		Lx := NewPrimeField(1)
		for j := 0; j < t; j++ {
			if i == j {
				continue
			}
			Lx = Lx.Mul(x.Sub(xs[j]).Div(xs[i].Sub(xs[j])))
		}
		res = res.Add(Lx.Mul(y))
	}
	return res
}

func Reconstruct(ShareHandles []io.Reader, xsHandles []io.Reader, outFile io.Writer, t int) {
	inBytes := make([][]byte, t)
	for i, file := range ShareHandles {
		data, _ := io.ReadAll(file)
		inBytes[i] = data
	}
	xsBytes := make([][]byte, t)
	for i, file := range xsHandles {
		data, _ := io.ReadAll(file)
		xsBytes[i] = data
	}
	dAShares := make([]*PrimeField, t)
	xs := make([]*PrimeField, t)
	for i := 0; i < t; i++ {
		dAShares[i] = &PrimeField{val: new(big.Int).SetBytes(inBytes[i])}
		xs[i] = &PrimeField{val: new(big.Int).SetBytes(xsBytes[i])}
	}

	secret := lagrangeRecons(xs, dAShares, t)
	outBytes := secret.ToBytes()
	outFile.Write(outBytes)
}
