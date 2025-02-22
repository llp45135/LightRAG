import unittest
import json
from lightrag.markdown_chunking import markdown_hierarchical_chunking

class TestMarkdownChunking(unittest.TestCase):

    def test_markdown_hierarchical_chunking(self):
        # Sample Markdown content
        markdown_content = """
# Chapter 1
Chapter 1 content
## Section 1.1
Content of section 1.1.
### Subsection 1.1.1
Content of subsection 1.1.1.
### Subsection 1.1.2
Content of subsection 1.1.2.
## Section 1.2
Content of section 1.2.
### Subsection 1.2.1
Content of subsection 1.2.1.
# Chapter 2
Content of chapter 2.
"""

        # Call the function
        result = markdown_hierarchical_chunking(markdown_content)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        


        # Check the result
        self.assertGreater(len(result), 0)  # Ensure that chunks are created
        self.assertEqual(result[0]['heading_text'], "# Chapter 1")  # Check the first chunk's heading
        self.assertEqual(result[1]['heading_text'], "## Section 1.1")  # Check the second chunk's heading

if __name__ == '__main__':
    unittest.main() 