import subprocess
import sys
import os
import time
import psutil
import hashlib
from typing import Tuple, Dict

def start_prover(prover_path: str) -> subprocess.Popen:
    return subprocess.Popen([prover_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def read_prover_output(prover_process: subprocess.Popen, num_bytes: int) -> bytes:
    return prover_process.stdout.read(num_bytes)

def send_to_prover(prover_process: subprocess.Popen, data: bytes) -> None:
    prover_process.stdin.write(data)
    prover_process.stdin.flush()

def generate_input_data(N: int) -> bytes:
    input_data = b''
    for i in range(N):
        input_data += (i.to_bytes(8, 'little') * 8)
    return input_data

def measure_peak_memory_usage(proc: psutil.Process) -> int:
    peak_memory = 0
    try:
        while proc.poll() is None:
            mem = proc.memory_info().rss
            if mem > peak_memory:
                peak_memory = mem
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    return peak_memory

def verify_sha256_hashes(input_data: bytes, hash_results: bytes, N: int) -> bool:
    for i in range(N):
        chunk = input_data[i * 64:(i + 1) * 64]
        expected_hash = hashlib.sha256(chunk).digest()
        actual_hash = hash_results[i * 32:(i + 1) * 32]
        if expected_hash != actual_hash:
            return False
    return True

def main(prover_path: str, verifier_path: str, circuit_path: str) -> None:
    # Start benchmarking
    benchmark: Dict[str, float] = {}

    # Step 1: Start the prover program
    prover_process = start_prover(prover_path)
    prover_pid = prover_process.pid
    prover_psutil = psutil.Process(prover_pid)

    # Step 2: Prover sends `N` to SPJ's stdout
    N = int.from_bytes(read_prover_output(prover_process, 8), 'little')
    print(f"Prover sent N: {N}")

    # Step 3: SPJ reads your serialized circuit from your provided `circuit` file
    with open(circuit_path, 'rb') as f:
        serialized_circuit_bytes = f.read()

    # Step 4: SPJ sends to prover's stdin
    send_to_prover(prover_process, serialized_circuit_bytes)

    # Measure setup time
    setup_start_time = time.time()
    setup_finished = read_prover_output(prover_process, 16)
    setup_time = time.time() - setup_start_time
    benchmark['setup_time'] = setup_time
    print(f"Prover setup finished message: {setup_finished}")

    # Step 6: Generate input data based on N and send to prover's stdin
    input_data = generate_input_data(N)
    send_to_prover(prover_process, input_data)

    # Measure witness generation time
    witness_start_time = time.time()
    hash_results = read_prover_output(prover_process, 32 * N)
    witness_generated = read_prover_output(prover_process, 24)
    witness_time = time.time() - witness_start_time
    benchmark['witness_generation_time'] = witness_time
    print(f"Prover hash results: {hash_results.hex()}")
    print(f"Prover witness generated message: {witness_generated}")

    # Read proof bytes until prover process terminates
    proof_bytes = b''
    while prover_process.poll() is None:
        proof_bytes += prover_process.stdout.read()

    proof_time = time.time() - witness_start_time
    benchmark['proof_generation_time'] = proof_time
    benchmark['proof_size'] = len(proof_bytes)
    print(f"Prover proof bytes: {proof_bytes.hex()}")

    # Step 10: Prover program quits
    prover_process.stdin.close()
    prover_process.stdout.close()
    prover_process.wait()

    # Measure peak memory usage
    peak_memory_usage = measure_peak_memory_usage(prover_psutil)
    benchmark['peak_memory'] = peak_memory_usage

    # Verify SHA256 hash results
    spj_start_time = time.time()
    if not verify_sha256_hashes(input_data, hash_results, N):
        print("SHA256 hash verification failed.")
        return
    spj_processing_time = time.time() - spj_start_time
    benchmark['spj_processing_time'] = spj_processing_time

    # Step 11: SPJ invokes the verifier and sends `YOUR PROOF BYTES` to verifier's stdin
    verifier_start_time = time.time()
    verifier_process = subprocess.Popen([verifier_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    verifier_output, _ = verifier_process.communicate(input=proof_bytes)
    verification_time = time.time() - verifier_start_time
    benchmark['verification_time'] = verification_time

    # Verifier outputs a byte to stdout, byte `00` represents false, byte `ff` represents true
    if verifier_output == b'\xff':
        print("Verification successful: True")
    else:
        print("Verification failed: False")

    # Output benchmark details
    print("\nBenchmark Details:")
    for key, value in benchmark.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python spj.py <path_to_prover_executable> <path_to_verifier_executable> <path_to_circuit_file>")
        sys.exit(1)
    
    prover_path = sys.argv[1]
    verifier_path = sys.argv[2]
    circuit_path = sys.argv[3]
    
    main(prover_path, verifier_path, circuit_path)
