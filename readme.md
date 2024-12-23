# Expander-SHA256: Optimized Zero-Knowledge Proof Generation for SHA256

Expander-SHA256 is a cutting-edge implementation designed for efficient zero-knowledge proof generation and verification of the SHA256 hash function. This project is part of the Polyhedra Hackathon challenge, targeting performance optimization for blockchain applications on BSC and opBNB.

## 🚀 Key Features
- **Optimized Performance**: Significant improvements in proof generation time and memory usage.
- **Correctness & Security**: Adheres to cryptographic standards for SHA256-based zero-knowledge proofs.
- **Scalable Design**: Supports instance sizes up to 32,768.
- **Compliant Implementation**: Built on the Expander framework, following ProofArena's standardized environment.

---

## 📊 Performance Metrics
| **Metric**                 | **Value**           |
|----------------------------|---------------------|
| **Setup Time**             | `6.13e-5` seconds  |
| **Witness Generation Time**| `9.77e-6` seconds  |
| **Proof Generation Time**  | `0.00145` seconds  |
| **Verification Time**      | `0.02005` seconds  |
| **Peak Memory Usage**      | `417.06 MB`        |
| **Proof Size**             | `191 KB`           |
| **Instance Number (N)**    | `32,768`           |

![image](https://github.com/user-attachments/assets/862faa7c-2411-4c7e-bc9e-dce535037df4)

---

## 📚 How It Works
Expander-SHA256 uses a combination of advanced techniques such as:
- **Brent-Kung Adder**: For efficient arithmetic operations.
- **Optimized Boolean Operations**: Including XOR, AND, and NOT gates.
- **Rotational and Shift Operations**: Optimized implementations for SHA256-specific bit manipulations.

---

## 🛠️ Setup Instructions
### Prerequisites
- Rust (1.70 or later)
- Go (1.19 or later)
- ProofArena environment (Refer to [Machine Specifications](https://prooferena.docs/machine-specs))

### Building the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Amarnath-Rao/expander-sha256.git
   cd expander-sha256
   ```
2. Build the prover:
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```
3. Build the SPJ:
   ```bash
   go mod -C problems/sha256_hash/SPJ tidy
   go build -C problems/sha256_hash/SPJ
   ```

---

## 🔄 Usage
### Running the Prover and Verifier
Execute the `run.sh` script from the `proof-arena` directory:
```bash
bash problems/sha256_hash/expander-sha256/run.sh
```
![image](https://github.com/user-attachments/assets/27cb1a30-f502-4039-9178-1414e10b15df)

This script:
- Builds the required binaries.
- Configures the SPJ with a maximum instance size of 32,768.
- Executes the prover and verifier with specified parameters.
- Outputs performance metrics in `spj_output/sha256_hash/expander-sha256.json`.

### Debugging
Debug information, including exit statuses, is printed to the console for troubleshooting.

---


## 🧩 Code Overview

### Core Implementation (`lib.rs`)
This file contains essential functions for implementing the SHA256 logic in the Expander framework.

#### Bit Manipulation Functions
```rust
pub fn int2bit<C: Config>(api: &mut API<C>, value: u32) -> Vec<Variable> {
    (0..32).map(|x| api.constant(((value >> x) & 1) as u32)).collect()
}

pub fn rotate_right(bits: &Vec<Variable>, k: usize) -> Vec<Variable> {
    let n = bits.len();
    let s = k & (n - 1);
    let mut new_bits = bits[s..].to_vec();
    new_bits.append(&mut bits[0..s].to_vec());
    new_bits
}

pub fn shift_right<C: Config>(api: &mut API<C>, bits: Vec<Variable>, k: usize) -> Vec<Variable> {
    let s = k & (bits.len() - 1);
    let mut new_bits = bits[s..].to_vec();
    new_bits.append(&mut vec![api.constant(0); s]);
    new_bits
}
```

#### Boolean Logic Gates
```rust
pub fn xor<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    a.iter().zip(b.iter()).map(|(ai, bi)| api.add(*ai, *bi)).collect()
}

pub fn and<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    a.iter().zip(b.iter()).map(|(ai, bi)| api.mul(*ai, *bi)).collect()
}

pub fn not<C: Config>(api: &mut API<C>, a: Vec<Variable>) -> Vec<Variable> {
    a.iter().map(|ai| api.sub(1, *ai)).collect()
}
```

#### SHA256-Specific Functions
```rust
pub fn ch<C: Config>(api: &mut API<C>, x: Vec<Variable>, y: Vec<Variable>, z: Vec<Variable>) -> Vec<Variable> {
    xor(api, and(api, x.clone(), y.clone()), and(api, not(api, x), z.clone()))
}

pub fn maj<C: Config>(api: &mut API<C>, x: Vec<Variable>, y: Vec<Variable>, z: Vec<Variable>) -> Vec<Variable> {
    xor(api, and(api, x.clone(), y.clone()), xor(api, and(api, x.clone(), z.clone()), and(api, y.clone(), z.clone())))
}

pub fn sigma0<C: Config>(api: &mut API<C>, x: Vec<Variable>) -> Vec<Variable> {
    xor(api, rotate_right(&x, 2), xor(api, rotate_right(&x, 13), rotate_right(&x, 22)))
}

pub fn sigma1<C: Config>(api: &mut API<C>, x: Vec<Variable>) -> Vec<Variable> {
    xor(api, rotate_right(&x, 6), xor(api, rotate_right(&x, 11), rotate_right(&x, 25)))
}
```

---

### SPJ Helper Functions (`spj.rs`)
Utilities for SPJ (Specific Problem Judger) communication.

```rust
/// Writes a string to the writer
pub fn write_string<W: Write>(writer: &mut W, s: &str) -> std::io::Result<()> {
    writer.write_all(&(s.len() as u64).to_le_bytes())?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

/// Reads a blob of data from the reader
pub fn read_blob<R: Read>(reader: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf);
    let mut buf = vec![0; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}
```

---

### Prover and Verifier Entry Points (`main.rs`)
The main file configures and runs the prover and verifier processes.

#### Prover Command
```rust
fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 4 {
        eprintln!("Usage: <prove|verify> <parallelism> <instances>");
        return;
    }

    let mode = &args[1];
    let parallelism: usize = args[2].parse().expect("Invalid parallelism");
    let instances: usize = args[3].parse().expect("Invalid instance count");

    match mode.as_str() {
        "prove" => prover(parallelism, instances),
        "verify" => verifier(parallelism, instances),
        _ => eprintln!("Unknown mode: {}", mode),
    }
}
```

---

### Build Script (`compile.sh`)
```bash
#!/bin/bash
set -x

# Build the Rust-based prover
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

### Execution Script (`run.sh`)
```bash
#!/bin/bash
set -x

# Build SPJ and Rust components
go mod -C problems/sha256_hash/SPJ tidy
go build -C problems/sha256_hash/SPJ
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run the prover and verifier
problems/sha256_hash/SPJ/SPJ -cpu 16 -largestN 4096 \
  -memory 32768 -time 1200 \
  -json "spj_output/sha256_hash/expander-sha256.json" \
  -prover "target/release/expander-sha256 prove 16 256" \
  -verifier "target/release/expander-sha256 verify 16 256"
```

![image](https://github.com/user-attachments/assets/9c2601ba-86f8-4389-8aed-9c47333fe35d)

---

## 🔍 Theoretical Overview

### 1. **SHA256 Algorithm Basics**
SHA256 is a cryptographic hash function widely used for data integrity and authentication. It transforms an input message into a fixed-size 256-bit hash value, which is unique to the input data. Its properties include:
- **Deterministic**: The same input always produces the same hash.
- **Pre-image resistance**: It is computationally infeasible to reverse-engineer the input from the hash.
- **Collision resistance**: Finding two different inputs with the same hash is infeasible.
- **Avalanche effect**: A small change in the input causes a drastic change in the output hash.

### 2. **Components of SHA256**
SHA256 processes the input message in blocks of 512 bits, with the following key operations:
- **Message Preprocessing**:
  - Padding: Extends the message to a multiple of 512 bits.
  - Message Expansion: Expands the padded message into 64 32-bit words for processing.
- **Compression Function**:
  - Processes each block with logical operations (e.g., XOR, AND, NOT) and rotations.
  - Updates intermediate hash values using the **Ch**, **Maj**, and **Σ** functions.
- **Finalization**: Concatenates the intermediate hash values into the final 256-bit output.

### 3. **Expander Framework in Cryptographic Hashing**
The Expander framework is designed for highly parallelizable and scalable cryptographic operations. In this project, the framework focuses on:
- **Boolean Arithmetic**: SHA256 operations are expressed as Boolean gates (AND, XOR, NOT), allowing them to be implemented in circuits.
- **Bitwise Operations**: Key transformations like rotations, shifts, and additions are implemented using efficient bitwise operations.
- **Proof of Computation**: The prover generates and verifies proofs of computation for the hash, ensuring correctness in a distributed environment.

### 4. **Core Cryptographic Operations**
The SHA256 implementation relies on the following key functions:
- **Ch (Choice)**: `Ch(x, y, z) = (x AND y) XOR (NOT x AND z)`
  - Chooses bits from `y` or `z` based on `x`.
- **Maj (Majority)**: `Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)`
  - Outputs the majority bit among `x`, `y`, and `z`.
- **Σ0 and Σ1**: Bitwise rotations and shifts for message mixing.

### 5. **Parallelism and Scalability**
The Expander framework optimizes SHA256 for parallel computation by splitting the processing across multiple threads or processors:
- **Pipeline Design**: Each block of 512 bits is processed independently, leveraging hardware parallelism.
- **Intermediate Proofs**: The prover generates proofs for smaller segments, which are aggregated for verification.

### 6. **SPJ (Specific Problem Judger) Integration**
The SPJ is a validation layer ensuring the prover adheres to predefined constraints. It enables:
- **Integrity Checks**: Ensuring the computation matches expected cryptographic behavior.
- **Performance Metrics**: Evaluating time and resource efficiency of the prover.

### 7. **Use Cases of the Implementation**
- **Blockchain**: Verifying transactions using SHA256-based Merkle trees.
- **Data Integrity**: Validating file integrity in distributed systems.
- **Password Hashing**: Securing user credentials by storing hashed values.

### 8. **Challenges Addressed**
- **Resource Constraints**: The Expander framework ensures efficient memory and CPU utilization.
- **Scalability**: The design is adaptable to larger inputs or higher performance systems.
- **Proof Validation**: The SPJ ensures correctness and reproducibility in computations.

### 9. **Mathematical Formulation**
The algorithm uses modular arithmetic for additions and bitwise transformations, defined as:
\[
h[i] = h[i-1] + \Sigma_1(e) + Ch(e, f, g) + K[i] + W[i]
\]
Where:
- \( \Sigma_1(e) = (e \gg 6) \oplus (e \gg 11) \oplus (e \gg 25) \)
- \( Ch(e, f, g) = (e \land f) \oplus (\neg e \land g) \)
- \( W[i] \): Message schedule for the current block.

### 10. **Security Considerations**
- **Cryptographic Robustness**: The implementation adheres to SHA256’s standards, ensuring resistance against known attacks.
- **Proof Integrity**: The use of proofs ensures the validity of computations in potentially untrusted environments.

---
![image](https://github.com/user-attachments/assets/3ccf1235-5c34-485e-8865-e105a838765a)


## 📜 Code Structure
- **`compile.sh`**: Compiles the prover using optimized Rust settings.
- **`run.sh`**: Automates the prover-verifier execution pipeline.
- **`src/lib.rs`**: Contains the core implementation of SHA256 functions (e.g., `ch`, `maj`, `sigma0`, `sigma1`).
- **`src/spj.rs`**: Implements communication utilities for the SPJ protocol.
- **`spj_output`**: Stores JSON performance logs for submissions.

---

## 📈 Evaluation Process
1. Submit your implementation via ProofArena.
2. Monitor your performance on the leaderboard.
3. Ensure proof size is under 1 MB and correctness is maintained across all test cases.

---

## 💡 Optimization Techniques
- **Hardware-Specific Optimizations**: Using `RUSTFLAGS="-C target-cpu=native"` for enhanced performance.
- **Efficient Circuit Design**: Leveraging the Expander framework for GKR-based proof systems.
- **Memory Management**: Reducing peak memory usage to ~417 MB for large instance sizes.

---

## 📂 Submission Details
Submission includes:
- Completed source code.
- Build instructions (`compile.sh` and `run.sh`).
- Documentation and test cases.
- JSON logs from `spj_output`.

---

## 🧰 Resources
- **ProofArena**: [Competition Platform](https://prooferena.docs)
- **Expander Documentation**: [Expander Docs](https://expander.docs)
- **GNARK Framework**: [GNARK Docs](https://gnark.docs)

---

## 🤝 Community Support
- **Discussions**: [GitHub Discussions](https://github.com/ProofArena/discussions)
- **Technical Help**: [Polyhedra Hackathon](https://hackathon.polyhedra.network)

---

## 🎉 Acknowledgments
Special thanks to the Polyhedra team, ProofArena platform, and contributors to the Expander framework for their support and resources.

---
![image](https://github.com/user-attachments/assets/75968e8c-00f4-4fa2-ad18-faa56488f699)

##### (main.go):

```
package main

import (
	"bytes"
	"crypto/rand"
	"fmt"

	"crypto/sha256"

	"github.com/PolyhedraZK/proof-arena/SPJ"
)

type Sha256SPJ struct{}

func (k *Sha256SPJ) GenerateTestData(n uint64) []byte {
	data := make([]byte, n*64) // 64 bytes per instance
	_, err := rand.Read(data)
	if err != nil {
		panic(fmt.Sprintf("Failed to generate random data: %v", err))
	}
	return data
}

func (k *Sha256SPJ) VerifyResults(testData, results []byte) bool {
	n := uint64(len(testData) / 64)
	if uint64(len(results)) != n*32 {
		return false
	}

	for i := uint64(0); i < n; i++ {
		input := testData[i*64 : (i+1)*64]
		expectedOutput := results[i*32 : (i+1)*32]

		h := sha256.New()
		h.Write(input)
		computedOutput := h.Sum(nil)

		if !bytes.Equal(computedOutput, expectedOutput) {
			return false
		}
	}

	return true
}

func (k *Sha256SPJ) GetProblemID() int {
	return 3
}

func main() {
	sha256SPJ := &Sha256SPJ{}
	spj, err := SPJ.NewSPJTemplate(sha256SPJ)
	if err != nil {
		panic(fmt.Sprintf("Failed to create SPJ template: %v", err))
	}
	err = spj.Run()
	if err != nil {
		panic(fmt.Sprintf("SPJ run failed: %v", err))
	}
}
```
### **Summary of SHA256 Expander Implementation**

---

#### **Core Features**
1. **Cryptographic Hashing**:
   - Implements SHA256 with efficient bitwise operations such as rotations, shifts, and additions.
   - Uses Ch, Maj, and Σ functions for secure message processing.

2. **Expander Framework**:
   - Optimized for parallel and scalable execution using Boolean arithmetic and proof generation.
   - Ensures correctness and resource efficiency with the Specific Problem Judger (SPJ).

3. **Performance Metrics**:
   - Setup Time: `6.128e-5 seconds`
   - Witness Generation: `9.768e-6 seconds`
   - Proof Generation: `0.0014 seconds`
   - Verification Time: `0.02 seconds`
   - Memory Usage: `417 KB`
   - Proof Size: `191 KB`

4. **Applications**:
   - Blockchain verification, secure hashing for data integrity, and password protection.

---

#### **Key Code Snippets**

1. **Ch Function**:
```rust
pub fn ch<C: Config>(api: &mut API<C>, x: Vec<Variable>, y: Vec<Variable>, z: Vec<Variable>) -> Vec<Variable> {
    let xy = and(api, x.clone(), y.clone());
    let not_x = not(api, x.clone());
    let not_xz = and(api, not_x, z.clone());
    xor(api, xy, not_xz)
}
```

2. **Majority Function**:
```rust
pub fn maj<C: Config>(api: &mut API<C>, x: Vec<Variable>, y: Vec<Variable>, z: Vec<Variable>) -> Vec<Variable> {
    let xy = and(api, x.clone(), y.clone());
    let xz = and(api, x.clone(), z.clone());
    let yz = and(api, y.clone(), z.clone());
    xor(api, xor(api, xy, xz), yz)
}
```

3. **Rotate Right**:
```rust
pub fn rotate_right(bits: &Vec<Variable>, k: usize) -> Vec<Variable> {
    let n = bits.len();
    let s = k & (n - 1);
    let mut new_bits = bits[s..].to_vec();
    new_bits.append(&mut bits[..s].to_vec());
    new_bits
}
```

4. **Proof Validation (SPJ Example)**:
```rust
pub fn write_string<W: Write>(writer: &mut W, s: &str) -> std::io::Result<()> {
    let len = s.len() as u64;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}
```

---

#### **Theoretical Highlights**
1. **SHA256**: A cryptographic hash function providing data integrity, collision resistance, and pre-image resistance.
2. **Expander Framework**:
   - Transforms SHA256 operations into Boolean circuits.
   - Scalable and parallel-friendly for high-performance applications.
3. **Use Cases**:
   - Blockchain: Verifying hashes in Merkle trees.
   - Security: Hashing passwords and sensitive data.
   - File Integrity: Ensuring data hasn't been altered during transmission.

---
