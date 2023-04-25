from transformers import BertForSequenceClassification, BertTokenizer
import torch

# initialize our model and tokenizer
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# and we will place the processing of our input text into a function for easier prediction later
def sentiment(tokens):
    # get output logits from the model
    output = model(**tokens)
    # convert to probabilities
    probs = torch.nn.functional.softmax(output[0], dim=-1)
    # we will return the probability tensor (we will not need argmax until later)
    return probs

txt = """
I would like to get your all  thoughts on the bond yield increase this week.  I am not worried about the market downturn but the sudden increase in yields. On 2/16 the 10 year bonds yields increased by almost  9 percent and on 2/19 the yield increased by almost 5 percent.

Key Points from the CNBC Article:

* **The “taper tantrum” in 2013 was a sudden spike in Treasury yields due to market panic after the Federal Reserve announced that it would begin tapering its quantitative easing program.**
* **Major central banks around the world have cut interest rates to historic lows and launched unprecedented quantities of asset purchases in a bid to shore up the economy throughout the pandemic.**
* **However, the recent rise in yields suggests that some investors are starting to anticipate a tightening of policy sooner than anticipated to accommodate a potential rise in inflation.**

The recent rise in bond yields and U.S. inflation expectations has some investors wary that a repeat of the 2013 “taper tantrum” could be on the horizon.

The benchmark U.S. 10-year Treasury note climbed above 1.3% for the first time since February 2020 earlier this week, while the 30-year bond also hit its highest level for a year. Yields move inversely to bond prices.

Yields tend to rise in lockstep with inflation expectations, which have reached their highest levels in a decade in the U.S., powered by increased prospects of a large fiscal stimulus package, progress on vaccine rollouts and pent-up consumer demand.

The “taper tantrum” in 2013 was a sudden spike in Treasury yields due to market panic after the Federal Reserve announced that it would begin tapering its quantitative easing program.

Major central banks around the world have cut interest rates to historic lows and launched unprecedented quantities of asset purchases in a bid to shore up the economy throughout the pandemic. The Fed and others have maintained supportive tones in recent policy meetings, vowing to keep financial conditions loose as the global economy looks to emerge from the Covid-19 pandemic.

However, the recent rise in yields suggests that some investors are starting to anticipate a tightening of policy sooner than anticipated to accommodate a potential rise in inflation.

With central bank support removed, bonds usually fall in price which sends yields higher. This can also spill over into stock markets as higher interest rates means more debt servicing for firms, causing traders to reassess the investing environment.

“The supportive stance from policymakers will likely remain in place until the vaccines have paved a way to some return to normality,” said Shane Balkham, chief investment officer at Beaufort Investment, in a research note this week.

“However, there will be a risk of another ‘taper tantrum’ similar to the one we witnessed in 2013, and this is our main focus for 2021,” Balkham projected, should policymakers begin to unwind this stimulus.

Long-term bond yields in Japan and Europe followed U.S. Treasurys higher toward the end of the week as bondholders shifted their portfolios.

“The fear is that these assets are priced to perfection when the ECB and Fed might eventually taper,” said Sebastien Galy, senior macro strategist at Nordea Asset Management, in a research note entitled “Little taper tantrum.”

“The odds of tapering are helped in the United States by better retail sales after four months of disappointment and the expectation of large issuance from the $1.9 trillion fiscal package.”

Galy suggested the Fed would likely extend the duration on its asset purchases, moderating the upward momentum in inflation.

“Equity markets have reacted negatively to higher yield as it offers an alternative to the dividend yield and a higher discount to long-term cash flows, making them focus more on medium-term growth such as cyclicals” he said. Cyclicals are stocks whose performance tends to align with economic cycles.

Galy expects this process to be more marked in the second half of the year when economic growth picks up, increasing the potential for tapering.

## Tapering in the U.S., but not Europe

Allianz CEO Oliver Bäte told CNBC on Friday that there was a geographical divergence in how the German insurer is thinking about the prospect of interest rate hikes.

“One is Europe, where we continue to have financial repression, where the ECB continues to buy up to the max in order to minimize spreads between the north and the south — the strong balance sheets and the weak ones — and at some point somebody will have to pay the price for that, but in the short term I don’t see any spike in interest rates,” Bäte said, adding that the situation is different stateside.

“Because of the massive programs that have happened, the stimulus that is happening, the dollar being the world’s reserve currency, there is clearly a trend to stoke inflation and it is going to come. Again, I don’t know when and how, but the interest rates have been steepening and they should be steepening further.”

## Rising yields a ‘normal feature’

However, not all analysts are convinced that the rise in bond yields is material for markets. In a note Friday, Barclays Head of European Equity Strategy Emmanuel Cau suggested that rising bond yields were overdue, as they had been lagging the improving macroeconomic outlook for the second half of 2021, and said they were a “normal feature” of economic recovery.

“With the key drivers of inflation pointing up, the prospect of even more fiscal stimulus in the U.S. and pent up demand propelled by high excess savings, it seems right for bond yields to catch-up with other more advanced reflation trades,” Cau said, adding that central banks remain “firmly on hold” given the balance of risks.

He argued that the steepening yield curve is “typical at the early stages of the cycle,” and that so long as vaccine rollouts are successful, growth continues to tick upward and central banks remain cautious, reflationary moves across asset classes look “justified” and equities should be able to withstand higher rates.

“Of course, after the strong move of the last few weeks, equities could mark a pause as many sectors that have rallied with yields look overbought, like commodities and banks,” Cau said.

“But at this stage, we think rising yields are more a confirmation of the equity bull market than a threat, so dips should continue to be bought.”
"""

tokens = tokenizer.encode_plus(txt, add_special_tokens=False)

print(len(tokens['input_ids']))

print(type(tokens['input_ids']))

input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

print(input_ids[16:32])

start = 0
window_size = 512

# get the total length of our tokens
total_len = len(input_ids)

# initialize condition for our while loop to run
loop = True

# loop through and print out start/end positions
while loop:
    # the end position is simply the start + window_size
    end = start + window_size
    # if the end position is greater than the total length, make this our final iteration
    if end >= total_len:
        loop = False
        # and change our endpoint to the final token position
        end = total_len
    print(f"{start=}\n{end=}")
    # we need to move the window to the next 512 tokens
    start = end

# initialize probabilities list
probs_list = []

start = 0
window_size = 510  # we take 2 off here so that we can fit in our [CLS] and [SEP] tokens

loop = True

while loop:
    end = start + window_size
    if end >= total_len:
        loop = False
        end = total_len
    # (1) extract window from input_ids and attention_mask
    input_ids_chunk = input_ids[start:end]
    attention_mask_chunk = attention_mask[start:end]
    # (2) add [CLS] and [SEP]
    input_ids_chunk = [101] + input_ids_chunk + [102]
    attention_mask_chunk = [1] + attention_mask_chunk + [1]
    # (3) add padding upto window_size + 2 (512) tokens
    input_ids_chunk += [0] * (window_size - len(input_ids_chunk) + 2)
    attention_mask_chunk += [0] * (window_size - len(attention_mask_chunk) + 2)
    # (4) format into PyTorch tensors dictionary
    input_dict = {
        'input_ids': torch.Tensor([input_ids_chunk]).long(),
        'attention_mask': torch.Tensor([attention_mask_chunk]).int()
    }
    # (5) make logits prediction
    outputs = model(**input_dict)
    # (6) calculate softmax and append to list
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    probs_list.append(probs)

    start = end

# let's view the probabilities given
print(probs_list)

stacks = torch.stack(probs_list)
print(stacks)

shape = stacks.shape
print(shape)

#When we try to resize our tensor, we will receive this RuntimeError telling us that we cannot resize variables that
#require grad. What this is referring to is the gradient updates of our model tensors during training. PyTorch cannot
#calculate gradients for tensors that have been reshaped. Fortunately, we don't actually want to use this tensor during
#any training, so we can use the torch.no_grad() namespace to tell PyTorch that we do not want to calculate any gradients.


with torch.no_grad():
    # we must include our stacks operation in here too
    stacks = torch.stack(probs_list)
    # now resize
    stacks = stacks.resize_(stacks.shape[0], stacks.shape[2])
    # finally, we can calculate the mean value for each sentiment class
    mean = stacks.mean(dim=0)

print(mean)

print(torch.argmax(mean).item())