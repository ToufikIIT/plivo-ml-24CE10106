import tokenizer as tokenizer_mod

text = open("../data/train_corpus.txt", encoding="utf-8").read()
merges = tokenizer_mod.train_bpe(text, num_merges=256)   
tok = tokenizer_mod.BPETokenizer(merges)
tok.save()
print("vocab_size:", tok.vocab_size)

sample = text[:20000]
ids = tok.encode(sample)
assert tok.decode(ids) == sample, "round-trip FAILED"
print(f"round-trip OK — {len(sample.encode('utf-8'))} bytes -> {len(ids)} tokens "
      f"({len(sample.encode('utf-8'))/len(ids):.2f} bytes/token)")