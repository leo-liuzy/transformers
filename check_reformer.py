from transformers.models.reformer import ReformerModelWithLMHead, ReformerTokenizer, ReformerConfig
from transformers.models.reformer.modeling_reformer import LSHSelfAttention, LocalSelfAttention
from pdb import set_trace as bp
import torch
import argparse
import numpy as np

config = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
# config.random_buckets = True
# config.num_hashes = 64
model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment", config=config)
tok = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

short_inputs_lst = [
    "This evening, however, on coming out into the street, he became acutely aware of his fears.", 
    "On an exceptionally hot evening early in July a young man came out of the garret in which he lodged in S. Place and walked slowly, as though in hesitation, towards K. bridge.", 
    "The heat in the street was terrible: and the airlessness, the bustle and the plaster, scaffolding, bricks, and dust all about him, and that special Petersburg stench, so familiar to all who are unable to get out of town in summer--all worked painfully upon the young man’s already overwrought nerves.",
    "He was so badly dressed that even a man accustomed to shabbiness would have been ashamed to be seen in the street in such rags.", 
    "In that quarter of the town, however, scarcely any shortcoming in dress would have created surprise.", 
    "But there was such accumulated bitterness and contempt in the young man’s heart, that, in spite of all the fastidiousness of youth, he minded his rags least of all in the street.", 
    "A rouble and a half, and interest in advance, if you like!",
    "He only wanted a sling on his arm or a bandage on his finger to complete the impression of a man with a painful abscess or a broken arm.",
    "The light soon died away, but the look of suffering remained, and Zossimov, watching and studying his patient with all the zest of a young doctor beginning to practise, noticed in him no joy at the arrival of his mother and sister, but a sort of bitter, hidden determination to bear another hour or two of inevitable torture.",
]

long_inputs_lst = [
    '"Oh, Rodya, you wouldn’t believe," she began suddenly, in haste to answer his words to her, "how unhappy Dounia and I were yesterday! Now that it’s all over and done with and we are quite happy again--I can tell you. Fancy, we ran here almost straight from the train to embrace you and that woman--ah, here she is! Good morning, Nastasya!... She told us at once that you were lying in a high fever and had just run away from the doctor in delirium, and they were looking for you in the streets. You can’t imagine how we felt! I couldn’t help thinking of the tragic end of Lieutenant Potanchikov, a friend of your father’s--you can’t remember him, Rodya--who ran out in the same way in a high fever and fell into the well in the court-yard and they couldn’t pull him out till next day. Of course, we exaggerated things. We were on the point of rushing to find Pyotr Petrovitch to ask him to help.... Because we were alone, utterly alone," she said plaintively and stopped short, suddenly, recollecting it was still somewhat dangerous to speak of Pyotr Petrovitch, although "we are quite happy again."',
    'This was not because he was cowardly and abject, quite the contrary; but for some time past he had been in an overstrained irritable condition, verging on hypochondria. He had become so completely absorbed in himself, and isolated from his fellows that he dreaded meeting, not only his landlady, but anyone at all. He was crushed by poverty, but the anxieties of his position had of late ceased to weigh upon him. He had given up attending to matters of practical importance; he had lost all desire to do so. Nothing that any landlady could do had a real terror for him. But to be stopped on the stairs, to be forced to listen to her trivial, irrelevant gossip, to pestering demands for payment, threats and complaints, and to rack his brains for excuses, to prevaricate, to lie--no, rather than that, he would creep down the stairs like a cat and slip out unseen.',
    'The heat in the street was terrible: and the airlessness, the bustle and the plaster, scaffolding, bricks, and dust all about him, and that special Petersburg stench, so familiar to all who are unable to get out of town in summer--all worked painfully upon the young man’s already overwrought nerves. The insufferable stench from the pot-houses, which are particularly numerous in that part of the town, and the drunken men whom he met continually, although it was a working day, completed the revolting misery of the picture. An expression of the profoundest disgust gleamed for a moment in the young man’s refined face. He was, by the way, exceptionally handsome, above the average in height, slim, well-built, with beautiful dark eyes and dark brown hair. Soon he sank into deep thought, or more accurately speaking into a complete blankness of mind; he walked along not observing what was about him and not caring to observe it. From time to time, he would mutter something, from the habit of talking to himself, to which he had just confessed. At these moments he would become conscious that his ideas were sometimes in a tangle and that he was very weak; for two days he had scarcely tasted food.', 
    'And yet he was hastening to Svidrigaïlov; could he be expecting something _new_ from him, information, or means of escape? Men will catch at straws! Was it destiny or some instinct bringing them together? Perhaps it was only fatigue, despair; perhaps it was not Svidrigaïlov but some other whom he needed, and Svidrigaïlov had simply presented himself by chance. Sonia? But what should he go to Sonia for now? To beg her tears again? He was afraid of Sonia, too. Sonia stood before him as an irrevocable sentence. He must go his own way or hers. At that moment especially he did not feel equal to seeing her. No, would it not be better to try Svidrigaïlov? And he could not help inwardly owning that he had long felt that he must see him for some reason. But what could they have in common? Their very evil-doing could not be of the same kind. The man, moreover, was very unpleasant, evidently depraved, undoubtedly cunning and deceitful, possibly malignant. Such stories were told about him. It is true he was befriending Katerina Ivanovna\'s children, but who could tell with what motive and what it meant? The man always had some design, some project. There was another thought which had been continually hovering of late about Raskolnikov’s mind, and causing him great uneasiness. It was so painful that he made distinct efforts to get rid of it. He sometimes thought that Svidrigaïlov was dogging his footsteps. Svidrigaïlov had found out his secret and had had designs on Dounia. What if he had them still? Wasn\'t it practically certain that he had? And what if, having learnt his secret and so having gained power over him, he were to use it as a weapon against Dounia?'
]

print(f"No. hash: {config.num_hashes}")
chunk_len = config.local_attn_chunk_length

def set_chunk_length(model, length, atten_type = "lsh"):
    for layer in model.reformer.encoder.layers:
        # layer.attention.self_attention.chunk_length = length
        # if isinstance(layer.attention.self_attention, LSHSelfAttention):
            # layer.attention.self_attention.chunk_length = length
        if isinstance(layer.attention.self_attention, LocalSelfAttention):
            layer.attention.self_attention.chunk_length = length
        pass


for input_raw in short_inputs_lst:
    inputs = tok(input_raw, return_tensors="pt")
    
    seq_len = inputs['input_ids'].shape[-1]
    print(f"Seq_len({seq_len})")

    # assert seq_len > chunk_len
    print(f"Using LSHAttn: ")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

    bpd = loss / np.log(2)
    ppl = 2**bpd
    print(f"bpd: {bpd :.3f}")
    print(f"ppl: {ppl :.3f}")

    # hacky: force the model to use full attention 
    # set_chunk_length(model, seq_len * 32)
    # # bp()
    # print(f"Using FullAttn: ")
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # logits = outputs.logits

    # bpd = loss / np.log(2)
    # ppl = 2**bpd
    # print(f"bpd: {bpd :.3f}")
    # print(f"ppl: {ppl :.3f}")
    # set_chunk_length(model, chunk_len)
    print()
