#!/usr/bin/env python3
"""
T35: _prepare_inputs() implementation for vLLM backend.
This method builds input tensors from InputBatch for batched processing.
"""

# Insert this method into GgmlLLM class after _sample() method (line 367)
# and before stats() method (line 368)

_prepare_inputs_method = '''
    def _prepare_inputs(self, batch):
        """Prepare input tensors from InputBatch for batched processing (T35).
        
        Args:
            batch: InputBatch or dict containing:
                - request_ids: list of request identifiers
                - requests: dict mapping request_id -> {tokens, seq_len}
                - block_tables: dict mapping request_id -> list of block IDs
        
        Returns:
            dict with keys:
                - input_ids: np.ndarray [total_tokens] int32
                - positions: np.ndarray [total_tokens] int32  
                - seq_lens: list[int] [batch_size]
                - block_tables: np.ndarray [batch_size, max_blocks] int32
        """
        input_ids_list = []
        positions_list = []
        seq_lens = []
        block_tables_list = []
        
        max_blocks_per_seq = 0
        
        # Handle both InputBatch objects and raw dicts
        if hasattr(batch, 'request_ids'):
            # InputBatch object
            request_ids = batch.request_ids
            requests = batch.requests
            block_tables = batch.block_tables
        else:
            # Raw dict
            request_ids = batch.get('request_ids', [])
            requests = batch.get('requests', {})
            block_tables = batch.get('block_tables', {})
        
        for req_id in request_ids:
            req_data = requests.get(req_id, {})
            
            # Get tokens for this request
            tokens = req_data.get('tokens', [])
            seq_len = len(tokens)
            seq_lens.append(seq_len)
            
            # Extend input_ids and positions
            input_ids_list.extend(tokens)
            positions_list.extend(range(seq_len))
            
            # Get block table for this request
            bt = block_tables.get(req_id, [])
            block_tables_list.append(bt)
            max_blocks_per_seq = max(max_blocks_per_seq, len(bt))
        
        # Pad block tables to uniform width
        padded_block_tables = []
        for bt in block_tables_list:
            padded_bt = bt + [0] * (max_blocks_per_seq - len(bt))
            padded_block_tables.append(padded_bt)
        
        # Convert to numpy arrays (compatible with ctypes)
        input_ids = np.array(input_ids_list, dtype=np.int32)
        positions = np.array(positions_list, dtype=np.int32)
        block_tables = np.array(padded_block_tables, dtype=np.int32) if padded_block_tables else np.empty((0, 0), dtype=np.int32)
        
        return {
            "input_ids": input_ids,
            "positions": positions,
            "seq_lens": seq_lens,
            "block_tables": block_tables,
            "num_seqs": len(request_ids),
            "total_tokens": len(input_ids_list),
        }
'''

print(_prepare_inputs_method)
