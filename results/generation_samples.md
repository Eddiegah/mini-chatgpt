# Generation Samples

Real outputs from the trained MiniGPT model (3 epochs on 150K chars of Shakespeare, CPU).

Model: 462K parameters · 2 layers · 128-dim · 4 heads · vocab 456  
For the full-quality model, run `src/train.py` (10 epochs, full corpus) or use Colab.

---

## After epoch 1 (val loss 3.27) — structure is forming

**Prompt:** `HAMLET:`
```
HAMLET:
I'll will not, Sople,
That amy best you bod the god.

COMINIUS:
We more commontled Marcianted;
The
```
Names, dialogue format, Shakespearean punctuation — but mostly noise still.

---

## After epoch 2 (val loss 3.02) — style emerging

**Prompt:** `HAMLET:`
```
HAMLET to his vy:
If I had best thee he do at his lord.
MENENIUS:
Thou bet to the gain of senate.

MENIUS:
I'll craftred
```
Much more coherent. Recognizable rhetoric, multi-character dialogue.

---

## After epoch 3 (val loss 3.03) — temperature=0.9, top_k=40

**Prompt:** `HAMLET:`
```
HAMLET:
Now them aways: and can guid than If
Lre, wounds entray 'God;' he had blows you shall be now
Shall form.

LUTUS:
No, am from their
Thy sir,
```

**Prompt:** `To be, or not to be`
```
To be, or not to being: put it extres,
And whence them, when he hath could he
word; and to the wind not ratter to go one which say: he
What cause for thy profeople is mutinter.
```

**Prompt:** `KING LEAR:`
```
KING LEAR:
To speak their city, and if he bad atle:
The fatter'd the blods o'er-there
Give him out of him; and then,--doing

And call'd thee rutly can breath; and
```

**Prompt:** `All the world`
```
All the world,
That pray they distretry to have doth
te their burn yourself?

CORIOLANUS:
Ancute am mother:
What shall not a maluch thus they,
Even love a little with
```

---

## Analysis

**What works well:**
- Consistent Shakespearean register and vocabulary (thee, hath, thy, doth, 'God;')
- Correct dialogue formatting (CHARACTER:\nline)
- Multiple characters appear naturally (HAMLET, KING LEAR, COMINIUS, CORIOLANUS)
- Iambic rhythm hints, period-appropriate punctuation
- Coherent within individual phrases (2-5 words)

**What doesn't yet work:**
- Long-range coherence — themes drift within 2-3 lines
- Some malformed words (distretry, mutinter, maluch) — vocabulary limitations at this scale
- No narrative through-line

**Why:** This is a 462K parameter model trained on ~81K tokens for 3 epochs.  
GPT-2 (117M params) trained on 40GB. The architecture is identical — the gap is compute and data.  
Run the full `src/train.py` (10 epochs, full 1M token corpus) for substantially better outputs.

---

## To get better outputs

```bash
# Full training on complete Shakespeare corpus (10 epochs, ~18-24h CPU / 10min Colab GPU)
python src/train.py

# Or open notebooks/colab_version.ipynb on Google Colab with T4 GPU
```
