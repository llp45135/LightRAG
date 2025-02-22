# markdown_chunking.py
import re
from typing import List, Dict, Any

from lightrag.operate import chunking_by_token_size

def markdown_hierarchical_chunking(
    content: str,
    split_by_character: str | None = None,  # 忽略此参数，Markdown 结构化分块不依赖字符分割
    split_by_character_only: bool = False, # 忽略此参数
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """
    Markdown 结构化分块函数，按照章、节、小节层次进行分块，并处理超长内容。

    Args:
        content: Markdown 文档内容
        overlap_token_size: 块之间的 token 重叠大小
        max_token_size: 每个块的最大 token 数量
        tiktoken_model: 用于 token 化的模型名称

    Returns:
        分块结果列表，每个元素是一个字典，包含 "content", "tokens", "chunk_order_index", "structure_level", "heading_text"
    """
    chunks = []
    chunk_index = 0

    # 1. 按章节 (#) 分割
    chapter_pattern = re.compile(r"^#\s+(.+)", re.MULTILINE)
    chapter_splits = chapter_pattern.split(content)
    if len(chapter_splits) <= 1:
        chapter_chunks = [("", content)] # 没有章节标题，整个文档作为一个章节
    else:
        chapter_title = chapter_splits[1::2] # 章节标题
        chapter_content = chapter_splits[2::2] # 章节内容
        chapter_chunks = list(zip(chapter_title, chapter_content))

    for chapter_title_text, chapter_content_text in chapter_chunks:
        chapter_title_text = chapter_title_text.strip()
        if chapter_title_text:
            chapter_heading = f"# {chapter_title_text}"
        else:
            chapter_heading = "Document Top Level" # 没有章节标题的情况

        print(f"Processing Chapter: {chapter_heading}")  # Debugging statement

        # Extract only the relevant chapter content (up to the first section)
        section_pattern = re.compile(r"^##\s+(.+)", re.MULTILINE)
        section_match = section_pattern.search(chapter_content_text)
        if section_match:
            # Get content up to the first section heading
            chapter_content = chapter_content_text[:section_match.start()].strip()
        else:
            # If no sections, take all content
            chapter_content = chapter_content_text.strip()

        # Create a chunk for the chapter with its content
        base_chunk_meta = {
            "structure_level": "chapter",
            "heading_text": chapter_heading
        }
        
        # Collect section titles
        section_titles = []
        section_splits = section_pattern.split(chapter_content_text)
        if len(section_splits) > 1:
            section_titles = section_splits[1::2]  # Collect section titles

        # Include the chapter content in the chunk
        chunks.append({
            "tokens": 0,  # Placeholder for tokens, will be calculated later
            "content": f"下面的节标题是：{', '.join(section_titles)}\n内容是：{chapter_content}",  # Include chapter content
            "chunk_order_index": chunk_index,
            **base_chunk_meta
        })
        chunk_index += 1

        # 2. 在章节内按节 (##) 分割
        section_splits = section_pattern.split(chapter_content_text)

        if len(section_splits) <= 1:
            section_chunks = [("", section_splits[0])] # 没有节标题，整个章节内容作为一个节
        else:
            section_title = section_splits[1::2] # 节标题
            section_content = section_splits[2::2] # 节内容
            section_chunks = list(zip(section_title, section_content))

        for section_title_text, section_content_text in section_chunks:
            section_title_text = section_title_text.strip()
            if section_title_text:
                section_heading = f"## {section_title_text}"
            else:
                section_heading = chapter_heading # 如果节没有标题，继承章节标题

            print(f"Processing Section: {section_heading}")  # Debugging statement

            # Extract only the relevant section content (up to the first subsection)
            subsection_pattern = re.compile(r"^###\s+(.+)", re.MULTILINE)
            subsection_match = subsection_pattern.search(section_content_text)
            if subsection_match:
                # Get content up to the first subsection heading
                section_content = section_content_text[:subsection_match.start()].strip()
            else:
                # If no subsections, take all content
                section_content = section_content_text.strip()

            # Create a chunk for the section with its content
            base_chunk_meta = {
                "structure_level": "section",
                "heading_text": section_heading
            }
            subsection_titles = []  # Collect subsection titles
            subsection_splits = subsection_pattern.split(section_content_text)

            if len(subsection_splits) > 1:
                subsection_titles = subsection_splits[1::2]  # Collect subsection titles

            token_size_chunks = chunking_by_token_size(
                section_content,
                overlap_token_size=overlap_token_size,
                max_token_size=max_token_size,
                tiktoken_model=tiktoken_model
            )
            for chunk_data in token_size_chunks:
                chunks.append({
                    **chunk_data,
                    "chunk_order_index": chunk_index,
                    "content": f"下面的小节标题是：{', '.join(subsection_titles)}\n内容是：{section_content}\n本节内容属于{chapter_heading}",  # Include section content
                    **base_chunk_meta
                })
                chunk_index += 1

            # 3. 在节内按小节 (###) 分割
            for subsection_title_text, subsection_content_text in zip(subsection_titles, subsection_splits[2::2]):
                subsection_title_text = subsection_title_text.strip()
                if subsection_title_text:
                    subsection_heading = f"### {subsection_title_text}"
                else:
                    subsection_heading = section_heading  # 如果小节没有标题，继承节标题

                print(f"Processing Subsection: {subsection_heading}")  # Debugging statement

                # 4. 处理小节内容，如果仍然超长，则按 token 大小分割
                base_chunk_meta = {
                    "structure_level": "subsection",
                    "heading_text": subsection_heading
                }

                if subsection_content_text.strip():  # 避免空内容块
                    token_size_chunks = chunking_by_token_size(
                        subsection_content_text,
                        overlap_token_size=overlap_token_size,
                        max_token_size=max_token_size,
                        tiktoken_model=tiktoken_model
                    )
                    for chunk_data in token_size_chunks:
                        chunks.append({
                            **chunk_data,
                            "chunk_order_index": chunk_index,
                            "content": f"本小节内容属于{section_heading}",  # Include subsection content
                            **base_chunk_meta
                        })
                        chunk_index += 1

    return chunks

