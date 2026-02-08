import unittest
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models import ColumnDefinition, ColumnType, ColumnConstraints

class TestPersistence(unittest.TestCase):
    def test_column_definition_serialization(self):
        """Test that ColumnDefinition can be serialized to dict and back."""
        original_col = ColumnDefinition(
            name="Test Col",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="Generate a name",
            constraints=ColumnConstraints(min_length=5, max_length=100)
        )
        
        # Serialize
        data = original_col.model_dump()
        
        # Verify dict structure
        self.assertEqual(data['name'], "Test Col")
        self.assertEqual(data['type'], "Short Text")
        self.assertEqual(data['constraints']['min_length'], 5)
        
        # Deserialize
        restored_col = ColumnDefinition(**data)
        
        # Verify equality
        self.assertEqual(original_col.name, restored_col.name)
        self.assertEqual(original_col.type, restored_col.type)
        self.assertEqual(original_col.constraints.min_length, restored_col.constraints.min_length)

    def test_full_config_structure(self):
        """Test the structure used in save/load config."""
        cols = [
            ColumnDefinition(name="A", type=ColumnType.NUMERIC).model_dump(),
            ColumnDefinition(name="B", type=ColumnType.BOOLEAN).model_dump()
        ]
        
        config_data = {
            "model_id": "test-model",
            "num_rows": 100,
            "similarity_threshold": 0.9,
            "columns": cols
        }
        
        # Simulate saving to string
        json_str = json.dumps(config_data)
        
        # Simulate loading
        loaded_data = json.loads(json_str)
        
        self.assertEqual(loaded_data['model_id'], "test-model")
        self.assertEqual(len(loaded_data['columns']), 2)
        
        # Reconstruct objects
        col_objs = [ColumnDefinition(**c) for c in loaded_data['columns']]
        self.assertEqual(col_objs[0].name, "A")
        self.assertEqual(col_objs[0].type, ColumnType.NUMERIC)

if __name__ == '__main__':
    unittest.main()
