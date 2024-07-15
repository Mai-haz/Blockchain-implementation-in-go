# Blockchain-implementation-in-go


#### Overview
This project implements a simplified version of a blockchain-based airport management system. It includes functionalities for managing blocks, transactions, and peer-to-peer communication.

#### Functionality
- **Blockchain Management**: Implements basic blockchain operations including block validation, mining, and transaction handling.
- **Peer-to-Peer Communication**: Utilizes peer-to-peer networking for broadcasting blocks and transactions.
- **Merkle Tree**: Provides a Merkle tree implementation for efficient data verification and block integrity.

#### Setup Instructions
1. **Clone Repository**:
   ```
   git clone <repository_url>
   cd airport-management-system
   ```

2. **Install Dependencies**:
   Ensure you have Go installed. Dependencies can be managed using Go Modules.
   ```
   go mod tidy
   ```

3. **Run the Program**:
   ```
   go run main.go
   ```

#### Usage
- **Adding Transactions**:
  - Choose option 1 to add transactions.
  - Enter transaction value when prompted.

- **Mining Blocks**:
  - Choose option 2 to mine a new block.
  - Follow prompts to confirm and broadcast the block.

- **Displaying Blockchain**:
  - Choose option 3 to display all blocks in the blockchain.

- **Displaying Merkle Tree**:
  - Choose option 4 to display the Merkle tree of a specific block.

- **Starting P2P Network**:
  - Choose option 5 to start the peer-to-peer network functionality.

- **Modifying Blockchain**:
  - Choose option 6 to change the blockchain by adding or modifying blocks.

- **Verifying Blockchain**:
  - Choose option 7 to verify the integrity of the blockchain.

#### Contributors
- Mayhan Hazara

#### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
