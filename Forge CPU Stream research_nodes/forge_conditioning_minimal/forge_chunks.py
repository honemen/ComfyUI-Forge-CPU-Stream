"""Forge Neo chunk construction. Parser derived from Forge fixed commit 0c9273f.
Source: https://github.com/Haoming02/sd-webui-forge-classic/blob/0c9273f69dbe491246fb11c3d4ac70123099d393/backend/text_processing/parsing.py
"""
import re

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str, emphasis: str):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    if emphasis == "None":
        # interpret literally
        res = [[text, 1.0]]
    else:
        for m in re_attention.finditer(text):
            text = m.group(0)
            weight = m.group(1)

            if text.startswith("\\"):
                res.append([text[1:], 1.0])
            elif text == "(":
                round_brackets.append(len(res))
            elif text == "[":
                square_brackets.append(len(res))
            elif weight is not None and round_brackets:
                multiply_range(round_brackets.pop(), float(weight))
            elif text == ")" and round_brackets:
                multiply_range(round_brackets.pop(), round_bracket_multiplier)
            elif text == "]" and square_brackets:
                multiply_range(square_brackets.pop(), square_bracket_multiplier)
            else:
                parts = re.split(re_break, text)
                for i, part in enumerate(parts):
                    if i > 0:
                        res.append(["BREAK", -1])
                    res.append([part, 1.0])

        for pos in round_brackets:
            multiply_range(pos, round_bracket_multiplier)

        for pos in square_brackets:
            multiply_range(pos, square_bracket_multiplier)

        if len(res) == 0:
            res = [["", 1.0]]

        i = 0
        while i + 1 < len(res):
            if res[i][1] == res[i + 1][1]:
                res[i][0] += res[i + 1][0]
                res.pop(i + 1)
            else:
                i += 1

    return res


def build_chunks(segments, bos=49406, eos=49407, pad=49407, comma=267, capacity=75):
    """segments: [(content IDs, parsed text, weight)]. Exact ordinary Forge chunk rules."""
    chunks=[];ids=[];weights=[];last_comma=-1
    def finish():
        nonlocal ids,weights,last_comma
        missing=capacity-len(ids)
        row=[bos]+ids+[eos]*missing+[eos]
        ws=[1.0]+weights+[1.0]*missing+[1.0]
        if pad!=eos:
            end=row.index(eos);row[end+1:]=[pad]*(len(row)-end-1)
        chunks.append(list(zip(row,ws)))
        ids=[];weights=[];last_comma=-1
    for tokens,text,weight in segments:
        if text=='BREAK' and weight==-1:
            finish();continue
        for token in tokens:
            if token==comma:
                last_comma=len(ids)
            elif len(ids)==capacity and last_comma!=-1 and len(ids)-last_comma<=20:
                split=last_comma+1
                carry,carry_w=ids[split:],weights[split:]
                ids,weights=ids[:split],weights[:split]
                finish();ids,weights=carry,carry_w
            if len(ids)==capacity:finish()
            ids.append(token);weights.append(weight)
    if ids or not chunks:finish()
    return chunks


def tokenize_forge(text, sd_tokenizer):
    if sd_tokenizer.max_length!=77:
        raise ValueError('This adapter supports ordinary 77-token SDXL CLIP only')
    hf=sd_tokenizer.tokenizer
    parsed=parse_prompt_attention(text,'Original')
    tokenized=hf([t for t,w in parsed],truncation=False,add_special_tokens=False)['input_ids']
    segments=[(ids,t,w) for ids,(t,w) in zip(tokenized,parsed)]
    pairs=build_chunks(segments,bos=sd_tokenizer.start_token,eos=sd_tokenizer.end_token,
                       pad=sd_tokenizer.pad_token,comma=hf.get_vocab()[",</w>"])
    return pairs,parsed
