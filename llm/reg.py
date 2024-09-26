LIGAND_PAT = r"""'?<LIGAND>|<XYZ>|<eos>| ?\p{L}| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'?<LIGAND>|<XYZ>|<eos>| [^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,4}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""



ALL_PAT = r"""'?<LIGAND>|<MOL>|<XYZ>|<eos>| ?\p{L}| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""