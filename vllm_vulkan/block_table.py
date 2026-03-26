#!/usr/bin/env python3
"""
Vulkan Block Table Implementation for vLLM Plugin

Implements the BlockTable interface for paged KV cache management.
Matches vLLM's block_table.py API but uses Vulkan-compatible data structures.
"""

import numpy as np
import torch
from typing import List, Optional


class VulkanBlockTable:
    """
    Block table for managing paged KV cache blocks in Vulkan memory.
    
    Maps logical token positions to physical block IDs in the KV cache pool.
    Supports request lifecycle: allocate, free, move, swap.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        max_num_reqs: int = 128,
        max_num_blocks_per_req: int = 256,
        device: str = "cpu"  # Vulkan uses CUDA device string for torch tensors
    ):
        """
        Args:
            block_size: Number of tokens per KV cache block (typically 16)
            max_num_reqs: Maximum concurrent requests
            max_num_blocks_per_req: Maximum blocks per request (256 blocks × 16 tokens = 4096 context)
            device: Torch device for block table tensors
        """
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        
        # Block table: [max_num_reqs, max_num_blocks_per_req] int32
        # Each row contains block IDs for that request
        # -1 indicates unused block
        self.block_table = torch.full(
            (max_num_reqs, max_num_blocks_per_req),
            -1,
            dtype=torch.int32,
            device=device
        )
        
        # Number of blocks currently used per request
        self.num_blocks_per_req = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device
        )
        
        # Slot mapping: maps (request_idx, token_pos) → physical slot in KV cache
        # Computed on-demand during prepare_inputs
        self.slot_mapping = torch.zeros(
            max_num_reqs * max_num_blocks_per_req * block_size,
            dtype=torch.int64,
            device=device
        )
        
        # Free block pool management
        self.next_free_block = 0
        self.total_blocks = max_num_reqs * max_num_blocks_per_req
        
    def allocate_request(self, req_idx: int) -> List[int]:
        """
        Allocate blocks for a new request.
        
        Args:
            req_idx: Request index (0 to max_num_reqs-1)
            
        Returns:
            List of allocated block IDs
        """
        # Allocate first block for the request
        block_id = self.next_free_block
        self.next_free_block += 1
        
        if self.next_free_block > self.total_blocks:
            raise MemoryError(f"KV cache exhausted: {self.total_blocks} blocks allocated")
        
        # Record block in table
        self.block_table[req_idx, 0] = block_id
        self.num_blocks_per_req[req_idx] = 1
        
        return [block_id]
    
    def append_token(self, req_idx: int, token_pos: int) -> int:
        """
        Append a token to a request, allocating a new block if needed.
        
        Args:
            req_idx: Request index
            token_pos: Token position in the sequence
            
        Returns:
            Block ID where token is stored
        """
        num_blocks = self.num_blocks_per_req[req_idx].item()
        block_idx = token_pos // self.block_size
        
        # Check if we need a new block
        if block_idx >= num_blocks:
            if block_idx >= self.max_num_blocks_per_req:
                raise ValueError(f"Request {req_idx} exceeds max context length")
            
            # Allocate new block
            block_id = self.next_free_block
            self.next_free_block += 1
            
            if self.next_free_block > self.total_blocks:
                raise MemoryError(f"KV cache exhausted")
            
            self.block_table[req_idx, block_idx] = block_id
            self.num_blocks_per_req[req_idx] = block_idx + 1
            
            return block_id
        else:
            # Reuse existing block
            return self.block_table[req_idx, block_idx].item()
    
    def get_block_ids(self, req_idx: int) -> List[int]:
        """
        Get all block IDs for a request.
        
        Args:
            req_idx: Request index
            
        Returns:
            List of block IDs (up to num_blocks_per_req)
        """
        num_blocks = self.num_blocks_per_req[req_idx].item()
        return self.block_table[req_idx, :num_blocks].tolist()
    
    def get_slot_mapping(self, req_idx: int, token_pos: int) -> int:
        """
        Get physical slot index for a token.
        
        Args:
            req_idx: Request index
            token_pos: Token position in sequence
            
        Returns:
            Physical slot index in KV cache
        """
        block_idx = token_pos // self.block_size
        block_offset = token_pos % self.block_size
        
        block_id = self.block_table[req_idx, block_idx].item()
        
        # Physical slot = block_id * block_size + block_offset
        return block_id * self.block_size + block_offset
    
    def compute_slot_mappings_batched(
        self,
        req_indices: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute slot mappings for a batch of tokens.
        
        Args:
            req_indices: [num_tokens] request index for each token
            positions: [num_tokens] token position for each token
            
        Returns:
            [num_tokens] physical slot indices
        """
        num_tokens = len(req_indices)
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        
        for i in range(num_tokens):
            req_idx = req_indices[i].item()
            pos = positions[i].item()
            slot_mapping[i] = self.get_slot_mapping(req_idx, pos)
        
        return slot_mapping
    
    def move_request(self, src_req_idx: int, tgt_req_idx: int) -> None:
        """
        Move all blocks from one request to another (for reordering).
        
        Args:
            src_req_idx: Source request index
            tgt_req_idx: Target request index
        """
        num_blocks = self.num_blocks_per_req[src_req_idx].item()
        self.block_table[tgt_req_idx, :num_blocks] = self.block_table[src_req_idx, :num_blocks]
        self.num_blocks_per_req[tgt_req_idx] = num_blocks
        
        # Clear source
        self.num_blocks_per_req[src_req_idx] = 0
        
    def swap_requests(self, req_idx_a: int, req_idx_b: int) -> None:
        """
        Swap all blocks between two requests.
        
        Args:
            req_idx_a: First request index
            req_idx_b: Second request index
        """
        # Swap num_blocks
        num_blocks_a = self.num_blocks_per_req[req_idx_a].item()
        num_blocks_b = self.num_blocks_per_req[req_idx_b].item()
        
        self.num_blocks_per_req[req_idx_a] = num_blocks_b
        self.num_blocks_per_req[req_idx_b] = num_blocks_a
        
        # Swap block table rows
        temp = self.block_table[req_idx_a].clone()
        self.block_table[req_idx_a] = self.block_table[req_idx_b]
        self.block_table[req_idx_b] = temp
        
    def free_request(self, req_idx: int) -> None:
        """
        Free all blocks for a completed request.
        
        Args:
            req_idx: Request index to free
        """
        num_blocks = self.num_blocks_per_req[req_idx].item()
        
        # Reset block table entries
        self.block_table[req_idx, :num_blocks] = -1
        self.num_blocks_per_req[req_idx] = 0
        
        # Note: We don't actually reclaim blocks to next_free_block
        # to avoid fragmentation. In production, implement a proper
        # block pool with free list.
        
    def get_num_blocks(self, req_idx: int) -> int:
        """
        Get number of blocks allocated for a request.
        
        Args:
            req_idx: Request index
            
        Returns:
            Number of blocks
        """
        return self.num_blocks_per_req[req_idx].item()
    
    def get_total_allocated_blocks(self) -> int:
        """
        Get total number of blocks allocated across all requests.
        
        Returns:
            Total allocated blocks
        """
        return self.next_free_block
    
    def get_free_blocks(self) -> int:
        """
        Get number of free blocks remaining.
        
        Returns:
            Free block count
        """
        return self.total_blocks - self.next_free_block
    
    def to_numpy(self) -> np.ndarray:
        """
        Get block table as numpy array for debugging.
        
        Returns:
            [max_num_reqs, max_num_blocks_per_req] int32 array
        """
        return self.block_table.cpu().numpy()
    
    def debug_print(self) -> None:
        """
        Print block table state for debugging.
        """
        print(f"Block Table State:")
        print(f"  Total blocks: {self.total_blocks}")
        print(f"  Allocated: {self.next_free_block}")
        print(f"  Free: {self.get_free_blocks()}")
        print(f"  Block table (first 5 requests):")
        for i in range(min(5, self.max_num_reqs)):
            num_blocks = self.num_blocks_per_req[i].item()
            if num_blocks > 0:
                block_ids = self.get_block_ids(i)
                print(f"    Req {i}: {num_blocks} blocks → {block_ids}")


class VulkanBlockPool:
    """
    Manages the physical KV cache block pool in Vulkan memory.
    
    Allocates contiguous GPU memory for all blocks and tracks
    which blocks are in use via BlockTable.
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_kv_heads: int = 8,
        head_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu"
    ):
        """
        Args:
            num_blocks: Total number of KV cache blocks
            block_size: Tokens per block
            num_kv_heads: Number of KV heads
            head_size: Dimension per head
            dtype: Data type (float16 for memory efficiency)
            device: Torch device
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device
        
        # Calculate total memory needed
        # KV cache stores both K and V tensors
        # Shape: [num_blocks, 2 (K+V), num_kv_heads, block_size, head_size]
        self.total_elements = (
            num_blocks *
            2 *  # K and V
            num_kv_heads *
            block_size *
            head_size
        )
        self.total_bytes = self.total_elements * torch.tensor([], dtype=dtype).element_size()
        
        print(f"VulkanBlockPool: {num_blocks} blocks × {block_size} tokens = {num_blocks * block_size} max context")
        print(f"  Memory: {self.total_bytes / 1e6:.1f} MB ({num_kv_heads} heads × {head_size} dim × {dtype})")
        
        # Allocate contiguous GPU memory
        # Note: In production, allocate via Vulkan directly, not torch
        self.kv_cache = torch.empty(
            (num_blocks, 2, num_kv_heads, block_size, head_size),
            dtype=dtype,
            device=device
        )
        
        # Create block table for managing allocations
        self.block_table = VulkanBlockTable(
            block_size=block_size,
            max_num_reqs=128,
            max_num_blocks_per_req=num_blocks,
            device=device
        )
        
    def get_kv_block(
        self,
        block_id: int,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Get K and V tensors for a specific block and layer.
        
        Args:
            block_id: Physical block ID
            layer_idx: Transformer layer index
            
        Returns:
            Tuple of (K, V) tensors, each [num_kv_heads, block_size, head_size]
        """
        # In production, each layer has its own KV cache pool
        # For now, return the block (would need layer dimension in real impl)
        k = self.kv_cache[block_id, 0]  # [num_kv_heads, block_size, head_size]
        v = self.kv_cache[block_id, 1]  # [num_kv_heads, block_size, head_size]
        return k, v
    
    def write_kv_to_block(
        self,
        block_id: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> None:
        """
        Write K and V tensors to a block.
        
        Args:
            block_id: Physical block ID
            k: K tensor [num_kv_heads, block_size, head_size]
            v: V tensor [num_kv_heads, block_size, head_size]
        """
        self.kv_cache[block_id, 0] = k
        self.kv_cache[block_id, 1] = v
    
    def allocate_request(self, req_idx: int) -> List[int]:
        """
        Allocate blocks for a new request.
        
        Args:
            req_idx: Request index
            
        Returns:
            List of allocated block IDs
        """
        return self.block_table.allocate_request(req_idx)
    
    def free_request(self, req_idx: int) -> None:
        """
        Free blocks for a completed request.
        
        Args:
            req_idx: Request index
        """
        self.block_table.free_request(req_idx)


# Example usage and testing
if __name__ == "__main__":
    print("=== Vulkan Block Table Test ===")
    
    # Create block pool
    pool = VulkanBlockPool(
        num_blocks=256,
        block_size=16,
        num_kv_heads=8,
        head_size=128
    )
    
    # Allocate 3 requests
    print("\nAllocating requests...")
    req0_blocks = pool.allocate_request(0)
    print(f"Request 0: blocks {req0_blocks}")
    
    req1_blocks = pool.allocate_request(1)
    print(f"Request 1: blocks {req1_blocks}")
    
    req2_blocks = pool.allocate_request(2)
    print(f"Request 2: blocks {req2_blocks}")
    
    # Append tokens
    print("\nAppending tokens...")
    for i in range(20):
        block_id = pool.block_table.append_token(0, i)
        slot = pool.block_table.get_slot_mapping(0, i)
        if i % 16 == 0:
            print(f"  Token {i}: block {block_id}, slot {slot}")
    
    # Test move operation
    print("\nMoving request 0 → request 3...")
    pool.block_table.move_request(0, 3)
    print(f"Request 3 blocks: {pool.block_table.get_block_ids(3)}")
    
    # Test swap
    print("\nSwapping request 1 ↔ request 2...")
    pool.block_table.swap_requests(1, 2)
    print(f"Request 1 blocks: {pool.block_table.get_block_ids(1)}")
    print(f"Request 2 blocks: {pool.block_table.get_block_ids(2)}")
    
    # Free request
    print("\nFreeing request 0...")
    pool.free_request(0)
    print(f"Free blocks: {pool.block_table.get_free_blocks()}")
    
    # Debug print
    pool.block_table.debug_print()
    
    print("\n=== Test Complete ===")
