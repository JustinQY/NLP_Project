import random

def generate_prompt(characters, location="Central Perk", scenario="having coffee", seed_dialogue=None, lines=10):
    """
    自动生成剧本格式的 prompt
    - characters: List[str], 参与角色
    - location: str, 场景位置
    - seed_dialogue: Dict[str, str], 初始几句对白（可选）
    - lines: int, 希望生成的对白总行数
    """
    assert len(characters) >= 2, "至少要有两个角色参与对话"

    prompt = f"[Scene: {location}, {', '.join(characters)} are {scenario}.]\n\n"

    line_count = 0

    # 添加用户指定的种子对白
    if seed_dialogue:
        for speaker, line in seed_dialogue.items():
            prompt += f"{speaker}: {line.strip()}\n"
            line_count += 1

    # 补全剩下的空对白
    speaker_cycle = characters * ((lines // len(characters)) + 2)
    idx = 0

    while line_count < lines:
        speaker = speaker_cycle[idx % len(speaker_cycle)]
        idx += 1
        # ⚠️ 如果种子对白里已经给过这句话，我们只跳过“开头”，不是永远跳过这个人
        if seed_dialogue and speaker in seed_dialogue and line_count < len(seed_dialogue):
            continue
        prompt += f"{speaker}:\n"
        line_count += 1

    return prompt.strip()