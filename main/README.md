# Simple Secret Sharing

This project implements a threshold secret sharing scheme using polynomial interpolation over a prime field in Go. It's designed to securely split and reconstruct private keys used in cryptographic operations, specifically for ECDSA signatures on the secp256k1 curve.

## Features

- Secret sharing using Shamir's Secret Sharing scheme
- Implementation of prime field arithmetic
- Polynomial evaluation and interpolation
- Integration with secp256k1 curve for ECDSA signatures
- Threshold-based reconstruction of secrets

## How It Works

1. **Secret Splitting (Construct function):**
   - Takes a private key as input
   - Creates a polynomial of degree t-1 where f(0) is the secret
   - Generates n shares, where each share is a point (x, f(x)) on the polynomial

2. **Secret Reconstruction (Reconstruct function):**
   - Takes t shares as input
   - Uses Lagrange interpolation to reconstruct the original polynomial
   - Recovers the secret by evaluating f(0)

## Usage

The main functions provided by this library are:

```go
func Construct(dA *big.Int, outFilesValues []io.Writer, outFilesXs []io.Writer, n, t int)
func Reconstruct(ShareHandles []io.Reader, xsHandles []io.Reader, outFile io.Writer, t int)
```

- `Construct`: Splits a secret into n shares, requiring t shares for reconstruction.
- `Reconstruct`: Reconstructs the secret from t shares.

## Testing

The project includes a comprehensive test suite in `secret_share_test.go`. This test:

1. Generates an ECDSA key pair
2. Creates a signature
3. Splits the private key into shares
4. Reconstructs the private key from a subset of shares
5. Verifies the reconstructed key by checking the signature

To run the tests:

```
go test ./...
```

## Dependencies

- github.com/ethereum/go-ethereum/crypto/secp256k1: For elliptic curve operations

## Security Considerations

This implementation is for educational purposes and has not been audited for production use. In a real-world scenario, additional security measures should be implemented, such as:

- Secure random number generation
- Constant-time operations to prevent timing attacks
- Secure storage and transmission of shares

## License

This project is licensed under the terms specified in the LICENSE file.
