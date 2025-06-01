import json
import numpy as np

def group_by_course_blocks(text, max_length=2048) -> list[str]:
    """
    max_length 기준으로 청크로 나눔. 각 청크는 과목 단위로만 구성됨.
    """
    lines = text.strip().split('\n')
    course_blocks = []
    current_block = []

    for line in lines:
        if '.과목 코드:' in line:
            if current_block:  # 이전 블록 저장
                course_blocks.append("\n".join(current_block))
                current_block = []
        current_block.append(line)

    # 남은 라인도 포함
    if current_block:
        course_blocks.append("\n".join(current_block))

    # max_length 기준 청크 분리 (과목 단위 유지)
    chunks = []
    buffer = ""
    for block in course_blocks:
        if len(buffer) + len(block) + 1 > max_length:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
        buffer += block + "\n"
    if buffer:
        chunks.append(buffer.strip())

    return chunks