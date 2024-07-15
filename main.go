package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
)

type block struct {
	num                int
	hash               string
	hash_previous      string
	pointerToMerkle    *MerkleTree //pointer to the root node of merkle tree i.e root
	nonce              int
	transactional_data []*transaction
}

type transaction struct {
	Value     string
	signature string
}

type Blockchain struct {
	blocks []*block
}

type MerkleTree struct {
	RootNode *MerkleNode
}

type MerkleNode struct {
	Left  *MerkleNode
	Right *MerkleNode
	Data  []byte
}
type Node struct {
	ID                 int
	IP                 string
	Port               int
	Network            map[int]string // Keeps track of other nodes in the network (ID: IP:Port)
	KnownPeers         map[int]*Node  // Keeps track of known peers (ID: Node)
	recentTransactions []*transaction
	blockchain         *Blockchain
	//neighbour
}

func (ptrToBlock *block) calculateHash() string {

	//converting a data structure (such as an object) into a format that can be stored, transmitted, or reconstructed later
	//If the transactions were directly concatenated as strings without serialization, minor changes might not affect the hash, compromising the blockchain's security.
	jsonData, _ := json.Marshal(ptrToBlock.transactional_data)

	//concatenate the whole data of a block in blockchain

	blockData := string(jsonData) + strconv.Itoa(ptrToBlock.nonce) + string(ptrToBlock.pointerToMerkle.RootNode.Data) //the last thing will append all of the data of the transaction

	//we will use sha 256 to calculate the hash of the block

	hashObj := sha256.New()

	// writes the concatenated block data (converted to a byte slice) into the SHA256 hash object.
	hashObj.Write([]byte(blockData))

	//hashObj.Sum(nil) computes the final hash value and returns it as a byte slice.
	//hex.encodestring converts the byte slice hash value to a hexadecimal string.
	ptrToBlock.hash = hex.EncodeToString(hashObj.Sum(nil))

	return ptrToBlock.hash

}
func (ptrChain *Blockchain) MineBlock(trailingZeros int, ptrToBlock *block) (int, int) {

	min := 1000
	max := 9999
	//nonce := rand.Intn(max-min) + min
	x := strings.Repeat("0", trailingZeros)
	var nonce int
	//This loop continues until the hash of the block b does not have a prefix matching the string y (the target number of trailing zeros).
	//It's an implementation of a simple proof-of-work mechanism.
	for !strings.HasPrefix(ptrToBlock.hash, x) {
		nonce = rand.Intn(max-min) + min
		ptrToBlock.nonce = nonce
		ptrToBlock.calculateHash()
	}

	ptrChain.AddToBlock(ptrToBlock)
	fmt.Println("the block was mined")
	fmt.Printf("the nonce calculated was %d \n", nonce)
	return ptrToBlock.nonce, trailingZeros

}

func (ptrChain *Blockchain) AddToBlock(ptrToBlock *block) {

	prevBlock := ptrChain.blocks[len(ptrChain.blocks)-1]  //This line fetches the last block in the existing blockchain
	ptrToBlock.hash_previous = prevBlock.hash             //this line sets the hash of last block of blockchain to the prevhash of new block being added
	ptrChain.blocks = append(ptrChain.blocks, ptrToBlock) //It adds the new block to the end of the blockchain, extending the chain with the new block.
	fmt.Println("the transection was added to the block")
}

func CreteNewMerkleNode(left, right *MerkleNode, data []byte) *MerkleNode {

	newNode := &MerkleNode{}

	//This block checks if both the left and right child nodes are nil. If they are, it means this node is a leaf node without children.
	// In this case, it calculates the hash of the provided data using SHA256 and assigns the resulting hash to the Data field of the new node (newNode.Data).
	if left == nil && right == nil {
		hashvalue := sha256.Sum256(data)
		newNode.Data = hashvalue[:]

	} else { //If the node has child nodes (it's not a leaf node), it concatenates the hashes of the left and right child nodes (left.Data and right.Data) into prevHashes. Then it computes the hash of the concatenated hashes and assigns the resulting hash to the Data field of the new node.
		ChildHashes := append(left.Data, right.Data...)
		hashvalue := sha256.Sum256(ChildHashes)
		newNode.Data = hashvalue[:]

	}

	newNode.Left = left
	newNode.Right = right

	return newNode // Return the pointer to the dynamically allocated node
}
func NewBlock(t []*transaction) *block { //Function to create a new block
	//Initializing all the transactions
	Block := &block{
		transactional_data: t,
	}

	//Calculating root hash of merkel tree for the transactions of current block
	node := Block.HashTransactions()
	var p *MerkleTree = new(MerkleTree)
	p.RootNode = node
	Block.pointerToMerkle = p
	return Block
}
func CreateMerkleTree(data [][]byte) *MerkleTree { //Function to make the merkle tree
	var parents []MerkleNode

	//This specific line of code is used to ensure that the number of elements in the data slice is a power of 2. It pads the data slice with additional elements (copies of the last element) if the length of data is not a power of 2.
	//will append data until we reach the no of trnsactions of power 2
	for float64(int(math.Log2(float64(len(data))))) != math.Log2(float64(len(data))) {
		data = append(data, data[len(data)-1])
	}

	//create the leaf nodes for the initial transactions
	for _, dat := range data {
		newNode := CreteNewMerkleNode(nil, nil, dat)
		parents = append(parents, *newNode)
	}

	//this loop will start building up the tree from the leaf nodes up unitl parent node
	for i := 0; i < int(math.Log2(float64(len(data)))); i++ {
		var children []MerkleNode

		for j := 0; j < len(parents); j += 2 {
			newNode := CreteNewMerkleNode(&parents[j], &parents[j+1], nil)
			children = append(children, *newNode)
		}

		parents = children
	}
	// this line of code is setting up the merkletree variable as an instance of a MerkleTree struct, initializing it with the root node of the constructed Merkle tree.
	//The root node becomes the starting point to access the entire Merkle tree structure.
	merkletree := MerkleTree{&parents[0]}

	return &merkletree
}

func (ptrToBlock *block) DisplayMerkelTree() {
	var txHashes [][]byte

	for _, tx := range ptrToBlock.transactional_data {
		txHashes = append(txHashes, tx.Serialize())
	}
	for _, tx := range txHashes {
		fmt.Printf("%x\n", tx)
	}
}

// Serialization refers to the process of converting a complex data structure or object into a format
// that can be easily stored, transmitted, or reconstructed later.
// This process involves converting the object into a linear stream of bytes, which can then be written to a file, sent over a network, or stored in a database.
func (ptrToBlock *transaction) Serialize() []byte { //function to serialize
	var res bytes.Buffer
	encoder := gob.NewEncoder(&res)

	err := encoder.Encode(ptrToBlock)

	Handle(err)
	return res.Bytes()
}

// this method processes the transactions within a block by serializing them, storing their hashes,
// constructs a Merkle tree using these hashes,
// and finally returns the root node of the resulting Merkle tree.
func (b *block) HashTransactions() *MerkleNode { //function to serialize the transactions
	var totalhashes [][]byte

	for _, hashval := range b.transactional_data {
		totalhashes = append(totalhashes, hashval.Serialize())
	}
	tree := CreateMerkleTree(totalhashes)
	return tree.RootNode
}

func Handle(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func (ptrToBlock *block) AddTransaction(item *transaction, signature string) []*transaction {
	// Set the signature for the new transaction
	item.signature = signature

	// Add the transaction to the list of transactions
	ptrToBlock.transactional_data = append(ptrToBlock.transactional_data, item)

	return ptrToBlock.transactional_data
}

func changeBlock(b *block, t *transaction, index int) {
	b.transactional_data[index] = t //Changing the block
	node := b.HashTransactions()    //Hashing the transactions again

	//Forming the tree again for new value of root node
	var p *MerkleTree = new(MerkleTree)
	p.RootNode = node
	b.pointerToMerkle = p
}

//This method iterates through each block in the blockchain,
// recalculates the hash of each block, and compares it to the stored hash.
//If any block's recalculated hash differs from the stored hash, it implies that the block's data might have been tampered with or corrupted, leading to the "Chain Invalid" message being printed.
// This helps detect potential issues with the integrity of the blockchain.

func (ptrTochain *Blockchain) verifyChain() {
	for _, block := range ptrTochain.blocks { //traversing the entire chain
		prevHash := block.hash           //storing curret blockHash
		newHash := block.calculateHash() //storing hash after recalculating

		if prevHash != newHash { //if recalculated hash not same, block has been changed
			fmt.Println("Chain Invalid")
		} else {
			fmt.Println("chain valid")
		}

	}
}
func Genesis(b *block) *block { //function to initialize the genesis block

	return NewBlock(b.transactional_data)
}

func (chain *Blockchain) DisplayBlocks() { //Displaying all the block data]
	for _, block := range chain.blocks {
		fmt.Printf("\nBlock Number: %d\n", block.num)
		fmt.Printf("Previous Hash: %s\n", block.hash_previous)
		for _, b := range block.transactional_data {
			fmt.Printf("Data in Block: %v\n", b.Value)
		}
		fmt.Printf("Hash: %s\n", block.hash)
		fmt.Printf("Nonce: %d\n", block.nonce)
		fmt.Printf("Root: %x\n", block.pointerToMerkle.RootNode.Data)
		fmt.Printf("------------------------------------------------------------------------\n\n")
	}
}
func DisplayMerkelTree(ptr *MerkleNode, space int) {
	if ptr == nil {
		return
	}

	space = space + 2

	DisplayMerkelTree(ptr.Right, space)

	fmt.Println()

	for i := 2; i < space; i++ {
		fmt.Print(" ")
	}

	fmt.Printf("%x\n", ptr.Data)

	DisplayMerkelTree(ptr.Left, space)

}
func (b *block) DisplayMerkel() {

	fmt.Println("-------------------Merkle Tree-----------------------")

	DisplayMerkelTree(b.pointerToMerkle.RootNode, 0)

	fmt.Println("-----------------------------------------------------")

}

var (
	bootstrapIP   = "127.0.0.1" // Replace with your bootstrap node's IP
	bootstrapPort = 5000        // Replace with your bootstrap node's port
	nodeCount     = 0           // Tracks the number of nodes in the network
	mutex         sync.Mutex    // Mutex for synchronizing access to the network map
)

func (n *Node) Initialize() {
	n.KnownPeers = make(map[int]*Node)
	// ... other initializations
}

// RegisterWithBootstrap registers a node with the bootstrap node
func (n *Node) RegisterWithBootstrap() {
	conn, err := net.Dial("tcp", bootstrapIP+":"+strconv.Itoa(bootstrapPort))
	if err != nil {
		fmt.Println("Failed to connect to bootstrap node:", err)
		return
	}
	defer conn.Close()

	msg := "REGISTER:" + n.IP + ":" + strconv.Itoa(n.Port)
	_, err = fmt.Fprintf(conn, msg)
	if err != nil {
		fmt.Println("Failed to register with bootstrap node:", err)
		return
	}
	fmt.Println("Registered with bootstrap node")
}
func (n *Node) HandleConnection(conn net.Conn) {
	defer conn.Close()

	// Handle incoming messages here
	// Example: Read incoming message, process it, etc.
}
func (n *Node) HandleIncomingConnections() {
	listener, err := net.Listen("tcp", n.IP+":"+strconv.Itoa(n.Port))
	if err != nil {
		fmt.Println("Failed to start server:", err)
		return
	}
	defer listener.Close()

	fmt.Println("Node", n.ID, "listening on port", n.Port)

	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Error accepting connection:", err)
			continue
		}
		fmt.Println("Accepted connection from", conn.RemoteAddr()) // Print remote address (peer's address)
		go n.HandleConnection(conn)
	}
}
func (n *Node) JoinNetwork() {
	// Connect to the bootstrap node
	bootstrapConn, err := net.Dial("tcp", bootstrapIP+":"+strconv.Itoa(bootstrapPort))
	if err != nil {
		fmt.Println("Failed to connect to bootstrap node:", err)
		return
	}
	defer bootstrapConn.Close()

	// Send request to bootstrap node to get information about existing nodes
	msg := "GETNODES"
	_, err = fmt.Fprintf(bootstrapConn, msg)
	if err != nil {
		fmt.Println("Failed to get nodes from bootstrap node:", err)
		return
	}

	// Handle response from bootstrap node to get other node's IP:Port
	// Establish connections with the received nodes using net.Dial
	// Update KnownPeers with the received nodes' information

}

var (
	wg     sync.WaitGroup
	doneCh = make(chan struct{})
	done   chan struct{}
)
var GlobalKnownPeers = make(map[int][]int)

// Create five nodes and establish connections
var nodes []*Node

func P2P() {

	//defer wg.Done()
	// Create the bootstrap node
	bootstrapNode := &Node{
		ID:   nodeCount,
		IP:   "127.0.0.1", // Replace with your bootstrap node's IP
		Port: 5000,        // Replace with your bootstrap node's port

	}
	nodeCount++

	// Start the bootstrap node's server to handle incoming connections
	go bootstrapNode.HandleIncomingConnections()

	// Connect the bootstrap node
	nodes = append(nodes, bootstrapNode)

	// Create four more nodes and establish connections
	for i := 0; i < 4; i++ {
		newNode := &Node{
			ID:   nodeCount,
			IP:   "127.0.0.1", // Replace with your node's IP
			Port: 5001 + i,    // Use different ports for each node
		}
		nodeCount++

		// Register with the bootstrap node
		newNode.RegisterWithBootstrap()

		// Start server to handle incoming connections
		go newNode.HandleIncomingConnections()

		// Join the network by establishing connections
		go newNode.JoinNetwork()

		nodes = append(nodes, newNode)

		for _, node := range nodes {
			fmt.Printf("The nodes we created inside the P2P are")
			fmt.Printf("Node ID: %d, IP: %s, Port: %d\n", node.ID, node.IP, node.Port)
		}

	}

	// Simulate connections between nodes (for demonstration)
	// Simulate connections between nodes (for demonstration)
	/*for i, node := range nodes {
		for j, peer := range nodes {
			if i != j { // Avoid connecting a node to itself
				// Simulate node connecting to other nodes
				node.Initialize()
				node.KnownPeers[j] = peer
				GlobalKnownPeers[j] = peer
			}
		}
	}*/
	for _, node := range nodes {
		node.Initialize()
	}
	// Assuming nodes is a slice of Node
	// Assuming nodes is a slice of Node
	for i := 0; i < len(nodes); i++ {
		for j := 0; j < len(nodes); j++ {
			if i != j {
				nodes[i].KnownPeers[j] = nodes[j]
				GlobalKnownPeers[i] = append(GlobalKnownPeers[i], j)
			}
		}
	}

	for _, node := range nodes {
		fmt.Printf(" From inside Node %d known peers: ", node.ID)
		for _, peer := range node.KnownPeers {
			fmt.Printf("%d ", peer.ID)
		}
		fmt.Println() // Move to the next line for the next node
	}

	// Keep the bootstrap node running
	select {
	//case doneCh <- struct{}{}:
	//default:
	}

}

var broadcastedTransactions = make(map[int][]*transaction)

func (n *Node) broadcastTransaction(newTransaction *transaction, sender *Node, broadcastedTransactions map[int][]*transaction, signature string) {
	fmt.Printf("Node %d received transaction from Node %d: %v\n", n.ID, sender.ID, newTransaction)

	// Check if the transaction is not already in the list
	if !isTransactionInList(newTransaction, broadcastedTransactions[n.ID]) {
		// Set the signature for the new transaction
		newTransaction.signature = signature

		// Forward the transaction to other peers
		for _, peerID := range GlobalKnownPeers[n.ID] {
			// Avoid sending the transaction back to the sender
			if peerID != sender.ID {
				// Get the peer node by ID
				peerNode := getNodeByID(peerID)

				if peerNode != nil {
					// Send the transaction to the neighbor
					fmt.Printf("Node %d is going to broadcast to Node %d\n", n.ID, peerID)

					// Add the transaction to the list of broadcasted transactions
					broadcastedTransactions[n.ID] = append(broadcastedTransactions[n.ID], newTransaction)

					peerNode.broadcastTransaction(newTransaction, n, broadcastedTransactions, signature)
				}
			}
		}
	}
}

func getNodeByID(nodeID int) *Node {
	// Assuming you have a slice of nodes named 'nodes'
	for _, node := range nodes {
		if node.ID == nodeID {
			return node
		}
	}
	return nil
}
func (n *Node) receiveTransaction(newTransaction *transaction, signature string) {
	// Check if the transaction is not already in the list
	if !isTransactionInList(newTransaction, n.recentTransactions) {
		// Set the signature for the new transaction
		newTransaction.signature = signature

		// Add the new transaction to the list
		n.recentTransactions = append(n.recentTransactions, newTransaction)

		// Broadcast the transaction to other nodes using flooding
		fmt.Printf("Node %d received and broadcasted transaction: %v\n", n.ID, newTransaction)
		//n.broadcastTransaction(newTransaction, n)
		n.broadcastTransaction(newTransaction, n, broadcastedTransactions, signature)
	}
}

func (n *Node) printRecentTransactions() {
	fmt.Printf("Recent transactions for Node %d:\n", n.ID)
	for _, txn := range n.recentTransactions {
		fmt.Printf("%+v\n", txn)
	}
}

func isTransactionInList(newTransaction *transaction, list []*transaction) bool {
	for _, tx := range list {
		if tx == newTransaction {
			return true
		}
	}
	return false
}

var broadcastedBlocks = make(map[int][]*block)

func (n *Node) broadcastBlock(newBlock *block, sender *Node, broadcastedBlocks map[int][]*block, nonce int, zeros int) {
	//fmt.Printf("hi 10 \n")
	fmt.Printf("Node %d received block from Node %d: %+v\n", n.ID, sender.ID, newBlock)

	// Check if the block is not already in the list
	if !isBlockInList(newBlock, broadcastedBlocks[n.ID]) {

		if isValidBlock(newBlock, nonce, zeros) {

			// Check if the block is not already in the list
			if !isBlockInList(newBlock, n.blockchain.blocks) {

				// Add the new block to the blockchain
				n.blockchain.blocks = append(n.blockchain.blocks, newBlock)
				//just did this GlobalLongestChain = n.blockchain.blocks
				// Prune transactions that are included in the mined block
				n.pruneTransactions(newBlock.transactional_data)

				// Check if the new block extends the longest chain
				if len(n.blockchain.blocks) > len(GlobalLongestChain) {
					// Set the longest chain to the new chain
					GlobalLongestChain = n.blockchain.blocks
				}
				// Forward the block to other peers
				for _, peerID := range GlobalKnownPeers[n.ID] {
					// Avoid sending the block back to the sender
					if peerID != sender.ID {
						// Get the peer node by ID
						peerNode := getNodeByID(peerID)

						if peerNode != nil {
							// Send the block to the neighbor
							fmt.Printf("Node %d is going to broadcast block to Node %d\n", n.ID, peerID)

							// Add the block to the list of broadcasted blocks
							// Add the block to the list of broadcasted blocks
							if broadcastedBlocks[n.ID] == nil {
								broadcastedBlocks[n.ID] = make([]*block, 0)
							}
							broadcastedBlocks[n.ID] = append(broadcastedBlocks[n.ID], newBlock)

							peerNode.broadcastBlock(newBlock, n, broadcastedBlocks, nonce, zeros)
						}
					}
				}
			}
		}
	}
}

var GlobalLongestChain []*block
var GlobalBlockchain = &Blockchain{blocks: make([]*block, 0)}

func (n *Node) receiveBlock(newBlock *block, nonce int, zeros int) {
	n.blockchain = GlobalBlockchain
	//fmt.Printf("hi 2 \n")
	// Validate the block and its transactions
	if isValidBlock(newBlock, nonce, zeros) {
		//fmt.Printf("hi 3 \n")
		// Check if the block is not already in the list
		if !isBlockInList(newBlock, n.blockchain.blocks) {
			// Add the new block to the blockchain
			//fmt.Printf("hi 4 \n")
			n.blockchain.blocks = append(n.blockchain.blocks, newBlock)
			//just did this GlobalLongestChain = n.blockchain.blocks
			// Prune transactions that are included in the mined block
			fmt.Printf("hi 5 \n")
			n.pruneTransactions(newBlock.transactional_data)
			//fmt.Printf("hi 6 \n")

			// Check if the new block extends the longest chain
			if len(n.blockchain.blocks) > len(GlobalLongestChain) {
				// Set the longest chain to the new chain
				//fmt.Printf("hi 7 \n")
				GlobalLongestChain = n.blockchain.blocks
				//fmt.Printf("hi 8 \n")
			}
			// Broadcast the block to other nodes using flooding
			fmt.Printf("Node %d received and broadcasted block: %+v\n", n.ID, newBlock)
			//fmt.Printf("hi 9 \n")
			n.broadcastBlock(newBlock, n, broadcastedBlocks, nonce, zeros)
			//fmt.Printf("hi 10 \n")

			/*
					// Notify neighbors about the new longest chain
					n.broadcastLongestChain(GlobalLongestChain)
				}

				// Broadcast the block to other nodes using flooding
				fmt.Printf("Node %d received and broadcasted block: %+v\n", n.ID, newBlock)
				n.broadcastBlock(newBlock, n, broadcastedBlocks)*/
		}
	}
}
func isValidBlock(newBlock *block, nonce int, zeros int) bool {
	// Check if the block's hash is valid
	//fmt.Printf("hi 31\n")
	if !isValidHash(newBlock.hash, zeros) {
		return false
	}
	//fmt.Printf("hi 32\n")
	// Check if the nonce is valid
	if !isValidNonce(newBlock, nonce, zeros) {
		return false
	}
	//fmt.Printf("hi 33\n")
	// Validate each transaction in the block
	for _, transaction := range newBlock.transactional_data {
		if !isValidTransaction(transaction) {
			return false
		}
	}

	// If all checks pass, the block is considered valid
	//fmt.Printf("hi 34\n")
	return true
}

// Example functions for validation
func isValidHash(hash string, leadingZeros int) bool {
	// Check if the hash has the required number of leading zeros
	//fmt.Printf(" we are inside the isvalidhash fuction to check if it has proper number of trailing zeros\n")
	result := strings.HasPrefix(hash, strings.Repeat("0", leadingZeros))
	//fmt.Println(result) // Print the result
	return result
}

func isValidNonce(newBlock *block, nonce int, trailingZeros int) bool {
	// Set the nonce of the block
	newBlock.nonce = nonce

	// Calculate the hash with the given nonce
	newBlock.calculateHash()

	// Create the target string with the specified number of trailing zeros
	target := strings.Repeat("0", trailingZeros)

	// Check if the hash has the required number of trailing zeros

	fmt.Printf(" we are inside the isvalidnonce fuction\n")
	result := strings.HasPrefix(newBlock.hash, target)
	fmt.Println(result) // Print the result
	return result
}

// ////////////////////////////////////////////////////////////////////
func isValidTransaction(transaction *transaction) bool {
	// Implement your transaction validation logic here

	// Check if the transaction is signed
	if !transaction.isSigned() {
		fmt.Printf("checking if the transection is signed\n")
		return false
	}

	// Check the validity of inputs and outputs
	/*if !isValidInputOutput(transaction) {
		return false
	}*/

	return true
}
func (t *transaction) isSigned() bool {
	return t.signature != ""
}

// isValidInputOutput checks the validity of inputs and outputs in a transaction.
/*func isValidInputOutput(transaction *transaction) bool {
	// Implement logic to check the validity of inputs and outputs
	// This might involve checking UTXOs, amounts, and other transaction details

	// Check if there are inputs and outputs
	if len(transaction.inputs) == 0 || len(transaction.outputs) == 0 {
		return false
	}

	// Check each input
	for _, input := range transaction.inputs {
		// For simplicity, this example assumes a boolean flag indicating whether the input is valid
		if !input.isValid() {
			return false
		}
	}

	// Check each output
	for _, output := range transaction.outputs {
		// For simplicity, this example assumes a boolean flag indicating whether the output is valid
		if !output.isValid() {
			return false
		}
	}

	// Add more specific checks as needed

	return true
}*/

// ////////////////////////////////////////////////////////////////////////////////////
func (n *Node) pruneTransactions(blockTransactions []*transaction) {
	// Prune transactions from the local list that are included in the mined block
	var prunedTransactions []*transaction

	for _, localTransaction := range n.recentTransactions {
		found := false
		for _, blockTransaction := range blockTransactions {
			if localTransaction == blockTransaction {
				found = true
				break
			}
		}

		if !found {
			prunedTransactions = append(prunedTransactions, localTransaction)
		}
	}

	// Update the local transaction list with pruned transactions
	n.recentTransactions = prunedTransactions
	fmt.Println("Pruned Transactions:")
	for _, txn := range n.recentTransactions {
		fmt.Printf("%+v\n", txn)
	}

}

/*func (n *Node) broadcastLongestChain(longestChain []*block) {
	// Broadcast the longest chain to other nodes using flooding
	for _, peerID := range GlobalKnownPeers[n.ID] {
		// Get the peer node by ID
		peerNode := getNodeByID(peerID)

		if peerNode != nil {
			// Send the longest chain to the neighbor
			fmt.Printf("Node %d is going to broadcast the longest chain to Node %d\n", n.ID, peerID)
			peerNode.receiveLongestChain(longestChain)
		}
	}
}

func (n *Node) receiveLongestChain(longestChain []*block) {
	// Receive the longest chain from a neighbor
	fmt.Printf("Node %d received the longest chain\n", n.ID)

	// Update the local blockchain with the received longest chain
	n.blockchain.blocks = longestChain
}*/

func (n *Node) printRecentBlocks() {
	fmt.Printf("Recent blocks for Node %d:\n", n.ID)
	for _, b := range n.blockchain.blocks {
		fmt.Printf("%+v\n", b)
	}
}
func areBlocksEqual(block1, block2 *block) bool {
	// Compare each field of the blocks
	return block1.num == block2.num &&
		block1.hash == block2.hash &&
		block1.hash_previous == block2.hash_previous &&
		areMerkleTreesEqual(block1.pointerToMerkle, block2.pointerToMerkle) &&
		block1.nonce == block2.nonce &&
		areTransactionsEqual(block1.transactional_data, block2.transactional_data)
}

func areMerkleTreesEqual(tree1, tree2 *MerkleTree) bool {
	// Compare the root hashes of the Merkle trees
	fmt.Printf("cheking if the root nodes are equal \n")
	if tree1.RootNode != tree2.RootNode {
		return false
	}

	// Recursively compare the left and right subtrees
	return areMerkleNodesEqual(tree1.RootNode, tree2.RootNode)
}

func areMerkleNodesEqual(node1, node2 *MerkleNode) bool {
	// If both nodes are nil, they are equal
	fmt.Printf("cheking if the nodes are equal \n")
	if node1 == nil && node2 == nil {
		return true
	}

	// If one of the nodes is nil and the other is not, they are not equal
	if node1 == nil || node2 == nil {
		return false
	}

	// Compare the hashes of the current nodes

	// Recursively compare the left and right children
	return areMerkleNodesEqual(node1.Left, node2.Left) && areMerkleNodesEqual(node1.Right, node2.Right)
}

func (t *transaction) isEqual(other *transaction) bool {

	fmt.Printf("cheking if the transections are equal \n")
	return t.Value == other.Value
}

func areTransactionsEqual(transactions1, transactions2 []*transaction) bool {
	// Compare each transaction in the slices
	if len(transactions1) != len(transactions2) {
		return false
	}

	for i, txn := range transactions1 {
		// Assuming transactions have an isEqual method to compare their content
		if i >= len(transactions2) || !txn.isEqual(transactions2[i]) {
			return false
		}
	}

	return true
}

func isBlockInList(newBlock *block, list []*block) bool {
	fmt.Printf("check if block is in list \n")
	//fmt.Printf("hi 41 \n")
	for _, b := range list {
		if areBlocksEqual(b, newBlock) {
			//fmt.Printf("hi 42 \n")
			return true
		}
		//fmt.Printf("hi 43 \n")
	}
	//fmt.Printf("hi 44 \n")
	return false
}

func main() {
	fmt.Printf("Hello mayhan\n")
	var choice int
	txn1 := &transaction{Value: "Transaction 1 data"}
	txn2 := &transaction{Value: "Transaction 2 data"}
	txn3 := &transaction{Value: "Transaction 3 data"}
	// txn4 := &transaction{Value: "Transaction 4 data"}
	// txn5 := &transaction{Value: "Transaction 5 data"}
	items := []*transaction{}
	block1 := &block{transactional_data: items}
	block1.num = 0
	txn1.signature = "signature"
	txn2.signature = "signature1"
	txn3.signature = "signature2"
	var signature string = "mayhan"
	block1.AddTransaction(txn1, signature)
	//block1.AddTransaction(txn2)

	block2 := &block{transactional_data: items}
	block2.num = 1
	block2.AddTransaction(txn1, signature)
	block2.AddTransaction(txn2, signature)
	block2.AddTransaction(txn3, signature)
	//block2.AddTransaction(txn2, signature)
	//block2.DisplayMerkel()
	//block2.AddTransaction(txn4)

	block3 := &block{transactional_data: items}
	block3.num = 2
	block3.AddTransaction(txn3, signature)

	gen := Genesis(block1)
	gen.hash_previous = "0"
	gen.nonce = 0

	gen.calculateHash() //pow

	chain := Blockchain{[]*block{gen}}

	//Mining first block

	block4 := NewBlock(block2.transactional_data)
	block4.num = 1
	chain.MineBlock(1, block4)

	//Mining second block
	block5 := NewBlock(block3.transactional_data)
	block5.num = 2
	chain.MineBlock(1, block5)

	fmt.Println("\n      Printing initial state of Block Chain:")
	chain.DisplayBlocks()
	block5.DisplayMerkel()
	// Run P2P in a goroutine
	/*doneCh = make(chan struct{})

	// Start P2P network setup in the background
	wg.Add(1)
	go P2P()

	// Wait for P2P setup to complete
	wg.Wait()

	// Continue with the main program
	fmt.Println("P2P setup completed. Continue with main program.")

	for nodeID, knownPeer := range GlobalKnownPeers {
		fmt.Printf("Node %d knows Node %d\n", nodeID, knownPeer.ID)
	}*/
	for {
		fmt.Println("\nMenu:")
		fmt.Println("1. Add Transaction")
		fmt.Println("2. Mine Block")

		fmt.Println("3. Display Blocks")
		fmt.Println("4. Display Merkle Tree")
		fmt.Println("5. Start the P2P Network functionality")
		fmt.Println("6. Change block chain")
		fmt.Println("7. Verify BlockChain")
		fmt.Println("8. Exit")
		fmt.Print("Press Enter to continue...")
		fmt.Scanln()
		fmt.Print("Enter your choice: ")

		fmt.Scanln(&choice)

		switch choice {
		case 1:
			var value string
			fmt.Print("Enter transaction value: ")
			reader := bufio.NewReader(os.Stdin)
			value, _ = reader.ReadString('\n')

			// Trim spaces and newline characters
			value = strings.TrimSpace(value)
			// Create a new transaction and add it to the pending transactions
			newTxn := &transaction{Value: value}
			/////////////////////////////////////////////////////////////////
			//Why block 5//////////////////////////////
			block5.AddTransaction(newTxn, signature)

			var selectedNode *Node

			// Assuming you have a node instance
			someNode := &Node{ID: 0, IP: "127.0.0.1", Port: 5000} //any node can send transection

			// Access the 'ID' field of the 'someNode' instance
			selectedNode = someNode
			//////////////////////////////////////////////////////////

			// Assuming selectedNode is a valid Node instance
			// Assuming selectedNode is a valid Node instance
			// Loop through GlobalKnownPeers and print node IDs and their known peers

			for nodeID, knownPeers := range GlobalKnownPeers {
				fmt.Printf("from main")
				fmt.Printf("Node %d knows: %v\n", nodeID, knownPeers)
			}

			// Broadcast the new transaction to the selected node
			selectedNode.receiveTransaction(newTxn, signature)
			selectedNode.printRecentTransactions()

		case 2:
			// Mine a new block
			// Example: chain.MineBlock(trailingZeros, pendingTransactions)

			/////////////////////////////////////////////////////////////////////////////////
			nonce, zeros := chain.MineBlock(1, block5) //why are we sending block5. It could be any block
			///////////////////////////////////////////////////////////////////////////

			// Assuming you have a node instance

			var selectedNode *Node
			someNode := &Node{ID: 0, IP: "127.0.0.1", Port: 5000}

			// Access the 'ID' field of the 'someNode' instance
			selectedNode = someNode
			// Assuming selectedNode is a valid Node instance
			// Assuming selectedNode is a valid Node instance
			// Loop through GlobalKnownPeers and print node IDs and their known peers

			for nodeID, knownPeers := range GlobalKnownPeers {
				fmt.Printf("from main")
				fmt.Printf("Node %d knows: %v\n", nodeID, knownPeers)
			}

			// Broadcast the new transaction to the selected node
			//fmt.Printf("hi 1 \n")
			selectedNode.receiveBlock(block5, nonce, zeros)

		case 3:
			// Display all blocks in the blockchain
			chain.DisplayBlocks()
		case 4:
			// Display Merkle Tree of a specific block
			block4.DisplayMerkel()
		case 5:
			go P2P()
			/*for nodeID, knownPeer := range GlobalKnownPeers {
				fmt.Printf("Node %d knows Node %d\n", nodeID, knownPeer.ID)
			}*/
		case 6:
			var index int
			var itemValue string

			fmt.Print("Enter index of the block: ")
			fmt.Scanln(&index)

			fmt.Print("Enter transaction item value: ")
			fmt.Scanln(&itemValue)

			// Create a new transaction with the provided item value
			item := &transaction{Value: itemValue}

			changeBlock(block3, item, index)
			chain.verifyChain()
		case 7:
			chain.verifyChain()
		case 8:
			fmt.Println("Exiting...")
			return
		default:
			fmt.Println("Invalid choice. Please enter a valid option.")
		}
	}
}
