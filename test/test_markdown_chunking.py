import unittest
import json
from lightrag.operate import chunking_markdown_hierarchical

class TestMarkdownChunking(unittest.TestCase):

    def test_markdown_hierarchical_chunking(self):
        # Sample Markdown content
        with open('test/kg8.md', 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Call the function
        result = chunking_markdown_hierarchical(markdown_content)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        


        # Check the result
        # self.assertGreater(len(result), 0)  # Ensure that chunks are created
        # self.assertEqual(result[0]['heading_text'], "# Chapter 1")  # Check the first chunk's heading
        # self.assertEqual(result[1]['heading_text'], "## Section 1.1")  # Check the second chunk's heading

if __name__ == '__main__':
    unittest.main() 