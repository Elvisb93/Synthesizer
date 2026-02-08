
import unittest
from unittest.mock import MagicMock
from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints
from core.controller import GeneratorController

class TestDuplication(unittest.TestCase):
    def test_allow_duplicates_constraint(self):
        # 1. Setup Controller
        config = GeneratorConfig(model_id="test-model", num_rows=10, max_retries=5)
        controller = GeneratorController()
        
        # Mock LLM Client
        mock_llm = MagicMock()
        # Sequence:
        # Row 1: Col A="VAL_A", Col B="VAL_B"
        # Row 2: Col A="VAL_A" (Dup allowed), Col B="VAL_B" (Dup rejected) -> Retry B="VAL_B" ...
        # We need enough "VAL_B"s to exhaust retries
        mock_llm.generate_completion.side_effect = [
            "VAL_A", "VAL_B",             # Row 1
            "VAL_A",                      # Row 2 Col A (Allowed)
            "VAL_B", "VAL_B", "VAL_B", "VAL_B", "VAL_B", "VAL_B" # Row 2 Col B (Rejected 6 times > 5 retries)
        ]
        controller.llm_client = mock_llm
        
        # 2. Define Columns
        col_a = ColumnDefinition(
            name="col_a",
            type=ColumnType.SHORT_TEXT,
            constraints=ColumnConstraints(allow_duplicates=True)
        )
        
        col_b = ColumnDefinition(
            name="col_b",
            type=ColumnType.SHORT_TEXT,
            constraints=ColumnConstraints(allow_duplicates=False)
        )
        
        controller.initialize(config, [col_a, col_b])
        controller.llm_client = mock_llm # Re-inject
        
        # 3. Generate Row 1
        row1 = controller.generate_row()
        self.assertIsNotNone(row1)
        self.assertEqual(row1.data["col_a"], "VAL_A")
        self.assertEqual(row1.data["col_b"], "VAL_B")
        
        # 4. Generate Row 2
        # Col A gets "VAL_A" (Duplicate) -> Should ACCEPT
        # Col B gets "VAL_B" (Duplicate) -> Should REJECT 5 times -> Fail
        row2 = controller.generate_row()
        self.assertIsNone(row2, "Row 2 should fail because col_b excludes duplicates")
        
        # 5. Now change col_b to allow duplicates and try again
        col_b.constraints.allow_duplicates = True
        controller.initialize(config, [self.copy_col(col_a), self.copy_col(col_b)]) # Re-init fresh validator
        
        mock_llm.generate_completion.side_effect = [
            "VAL_A", "VAL_B", # Row 1
            "VAL_A", "VAL_B"  # Row 2
        ]
        controller.llm_client = mock_llm
        
        # Row 1
        row1_new = controller.generate_row()
        self.assertIsNotNone(row1_new)
        
        # Row 2 (should pass now)
        row2_new = controller.generate_row()
        self.assertIsNotNone(row2_new, "Row 2 should pass when col_b allows duplicates")
        self.assertEqual(row2_new.data["col_a"], "VAL_A")
        self.assertEqual(row2_new.data["col_b"], "VAL_B")

    def copy_col(self, col):
        # Helper to clone since pydantic copy() might be shallow or weird with mutable defaults
        return ColumnDefinition(
            name=col.name,
            type=col.type,
            constraints=ColumnConstraints(
                allow_duplicates=col.constraints.allow_duplicates
            )
        )

if __name__ == '__main__':
    unittest.main()
