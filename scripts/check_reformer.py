import argparse
import torch
import numpy as np
from typing import Dict, Any

from transformers.models.reformer import ReformerModelWithLMHead, ReformerTokenizer, ReformerConfig
from transformers.models.reformer.modeling_reformer import LSHSelfAttention, LocalSelfAttention

from pdb import set_trace as bp

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

name2test_examples = {
    "long": long_inputs_lst, 
    "short": short_inputs_lst, 
}

def eval_and_print(model: ReformerModelWithLMHead, inputs: Dict[str, Any]):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss # reformer calculate the mean loss

    bpd = loss / np.log(2)
    ppl = 2**bpd
    print(f"bpd: {bpd :.3f}")
    print(f"ppl: {ppl :.3f}")


def set_chunk_length(model, length: int, atten_type: str = "none"):
    # this function essentially turn a self_attention module into a full attention
    # reformer do chunking when seq_lenth is greater than its set `chunk_length`
    # by default, all self-attention in reformer use chunking
    for layer in model.reformer.encoder.layers:
        if atten_type == "all":
            layer.attention.self_attention.chunk_length = length
        elif atten_type == "lsh":
            if isinstance(layer.attention.self_attention, LSHSelfAttention):
                layer.attention.self_attention.chunk_length = length
        elif atten_type == "local":
            if isinstance(layer.attention.self_attention, LocalSelfAttention):
                layer.attention.self_attention.chunk_length = length
        else:
            assert atten_type == "none", f"Attenion type '{atten_type}' not supported"
            pass

def encode(list_of_strings, pad_token_id=0, **kwargs):
    list_of_strings = [str.encode(string) if not isinstance(string, bytes) else string for string in list_of_strings]
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        assert isinstance(string, bytes)
        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks

def enwiki8_tokenizer(list_of_strings, pad_token_id=0, **kwargs):
    if isinstance(list_of_strings, str):
        list_of_strings = [list_of_strings]
    input_ids, attention_masks = encode(list_of_strings, pad_token_id=pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }


def main(args):
    config = ReformerConfig.from_pretrained(args.model_name_or_path)
    # storing default chunk length
    chunk_len = config.local_attn_chunk_length
    print(f"Default chunk length: {chunk_len}")
    print(f"Default no. buckets: {config.num_buckets}")
    print(f"Default no. hash: {config.num_hashes}")
    # updating config with commandline args
    if args.num_hashes is not None:
        config.num_hashes = args.num_hashes
    if args.num_buckets is not None:
        config.num_buckets = args.num_buckets
    config.random_buckets = args.random_bucket
    config.remove_attention_type = args.remove_attention_type
    print("Assigning random bucket to each vector" if config.random_buckets else "Using default (cross-polytope) LSH")

    model = ReformerModelWithLMHead.from_pretrained(args.model_name_or_path, config=config)
    tok = enwiki8_tokenizer if args.model_name_or_path == "google/reformer-enwik8" else ReformerTokenizer.from_pretrained(args.model_name_or_path)

    test_exmaples = name2test_examples[args.test_example_type]

    for input_raw in test_exmaples:
        inputs = tok(input_raw, return_tensors="pt")
        # bp()
        seq_len = inputs['input_ids'].shape[-1]
        print(f"Test example seq_len = {seq_len}")

        print(f"All SelfAttn use chunking: ")
        set_chunk_length(model, chunk_len, "all")
        eval_and_print(model, inputs)

        # hacky: force the model to use full attention by resetting chunk_length
        set_chunk_length(model, seq_len * 32, args.turn_to_full_attention)
        print(f"Use Full Attention in '{args.turn_to_full_attention}' SelfAttn: ")
        eval_and_print(model, inputs)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check reformer through serveral args"
    )
    # fmt: off 
    # google/reformer-enwik8
    parser.add_argument('--model_name_or_path', metavar="Model", type=str, default="google/reformer-crime-and-punishment", help='The model name or the path to the file', )

    parser.add_argument('--test_example_type', metavar="Test",  choices=["long", "short"], default="long", help='Choose long or short sequences to run reformer through')
    # args to set LSH
    parser.add_argument('--random_bucket', action="store_true", help='Whether replace LSH with a random bucket assignment')
    parser.add_argument('--num_hashes', metavar="LSH", type=int, default=None, help='reset a new num_hash for LSH')
    parser.add_argument('--num_buckets', metavar="LSH", type=int, nargs='+', default=None, help='reset a new num_bucket for LSH')
    # args to setting self_attention
    parser.add_argument('--turn_to_full_attention', metavar="SelfAtten", choices=["all", "none", "lsh", "local"], default="none", help='Choose which self-attention to turn to full attention (no chunking)')
    parser.add_argument('--remove_attention_type', metavar="SelfAtten", choices=["lsh", "local", "none", "all"], default="none", help='Choose lsh or local self_attention to remove from calculation')
    
    # fmt: on
    args = parser.parse_args()
    if args.num_buckets and len(args.num_buckets) == 1:
        args.num_buckets = args.num_buckets[0]
    main(args)
