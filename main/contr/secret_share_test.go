// generate a signature and test the secret sharing and reconstruction

package secret_sharing

import (
	"bytes"
	"fmt"
	"io"
	"math/big"
	"testing"
	"time"

	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha256"

	insecure_rand "math/rand"

	"github.com/ethereum/go-ethereum/crypto/secp256k1"
)

func generateKeyPair() (pubkey, privkey []byte) {
	key, err := ecdsa.GenerateKey(secp256k1.S256(), rand.Reader)
	if err != nil {
		panic(err)
	}
	pubkey = secp256k1.S256().Marshal(key.X, key.Y)

	privkey = make([]byte, 32)
	blob := key.D.Bytes()
	copy(privkey[32-len(blob):], blob)

	return pubkey, privkey
}
func TestSecretShare(t *testing.T) {
	pubkey, privkey := generateKeyPair()
	fmt.Println("pubkey:", pubkey)
	fmt.Println("privkey:", privkey)

	msg := []byte("hello")
	hash := sha256.Sum256(msg)
	sig, err := secp256k1.Sign(hash[:], privkey)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("signature:", sig)

	// check the signature
	recoveredPubkey, err := secp256k1.RecoverPubkey(hash[:], sig)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("recovered pubkey:", recoveredPubkey)

	if !bytes.Equal(recoveredPubkey, pubkey) {
		t.Fatal("recovered pubkey does not match")
	}

	// verify the signature
	fmt.Println("sig length:", len(sig))
	verified := secp256k1.VerifySignature(pubkey, hash[:], sig[:64])
	if !verified {
		t.Fatal("signature verification failed")
	}
	fmt.Println("signature verified")

	// secret sharing version
	N := 5
	T := 3
	buffer_sk := make([]*bytes.Buffer, N)
	buffer_xs := make([]*bytes.Buffer, N)
	for i := 0; i < N; i++ {
		buffer_sk[i] = new(bytes.Buffer)
		buffer_xs[i] = new(bytes.Buffer)
	}
	io_writers_sk := make([]io.Writer, N)
	io_writers_xs := make([]io.Writer, N)
	for i := 0; i < N; i++ {
		io_writers_sk[i] = buffer_sk[i]
		io_writers_xs[i] = buffer_xs[i]
	}
	sk_bigint := big.NewInt(0)
	sk_bigint.SetBytes(privkey)
	Construct(sk_bigint, io_writers_sk, io_writers_xs, N, T)

	// test the reconstruction of the secret
	io_readers_sk := make([]io.Reader, T)
	io_readers_xs := make([]io.Reader, T)

	// randomly select T shares
	r := insecure_rand.New(insecure_rand.NewSource(time.Now().UnixNano()))
	selected_indices := r.Perm(N)[:T]
	for i := 0; i < T; i++ {
		io_readers_sk[i] = bytes.NewReader(buffer_sk[selected_indices[i]].Bytes())
		io_readers_xs[i] = bytes.NewReader(buffer_xs[i].Bytes())
	}
	io_writer_sk := new(bytes.Buffer)
	Reconstruct(io_readers_sk, io_readers_xs, io_writer_sk, T)
	fmt.Println("reconstructed sk:", io_writer_sk.Bytes())
}
