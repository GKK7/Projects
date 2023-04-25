from transformers import BertForSequenceClassification, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

txt = """
Good morning, everyone. Dean Steinberg, thank you for your kind introduction. And thank you for your service to our country. I’m grateful for your contributions – not only during your time in government but here at SAIS.

I’m particularly glad to be at this institution. SAIS has one of the oldest and most extensive China studies programs in the country. In 1979, the United States established full diplomatic relations with the People’s Republic of China. Just two years after, your university leaders had their own talks with their Chinese counterparts. The goal was to see whether Johns Hopkins and Nanjing University could partner together to educate future leaders.

The result: the establishment of the Hopkins-Nanjing Center in 1986 – one of the first Western academic programs in modern China. This collaboration has been tested by the realities and complexities of our bilateral relationship. But I believe the students on this campus have served as a reminder of the respect that the American and Chinese people have for each other. And they demonstrate that people around the world can learn from one another if we communicate openly and honestly – even and especially when we disagree.

Since I began my career, the relationship between the United States and China has undergone a significant evolution. In the 1970s, our relationship was defined by rapprochement and gradual normalization. I watched President Nixon make his famous journey to China in 1972. And I heard our two countries begin to speak to each other again after decades of silence. In the years that followed, I saw China choose to implement market reforms and open itself to the global economy, driving an impressive rise into the second-largest economy in the world. Its development was supported by assistance from the World Bank and other international economic institutions. And the U.S. Congress and successive administrations played a major role in supporting China’s integration into global markets.

But in recent years, I’ve also seen China’s decision to pivot away from market reforms toward a more state-driven approach that has undercut its neighbors and countries across the world. This has come as China is striking a more confrontational posture toward the United States and our allies and partners – not only in the Indo-Pacific but also in Europe and other regions.

Today, we are at a critical time. The world is confronting the largest land war in Europe since World War II – just as it recovers from a once-in-a-century pandemic. Debt challenges are mounting for low- and middle-income countries. Some nations, including our own, have faced pressures on their economic and financial systems. And a U.N. report released last month indicates that the Earth is likely to cross a critical global warming threshold within the next decade – if no drastic action is taken.

Progress on these issues requires constructive engagement between the world’s two largest economies. Yet our relationship is clearly at a tense moment.

So today, I would like to discuss our economic relationship with China. My goal is to be clear and honest: to cut through the noise and speak to this essential relationship based on sober realities. 

The United States proceeds with confidence in its long-term economic strength. We remain the largest and most dynamic economy in the world. We also remain firm in our conviction to defend our values and national security. Within that context, we seek a constructive and fair economic relationship with China. Both countries need to be able to frankly discuss difficult issues. And we should work together, when possible, for the benefit of our countries and the world.

Our economic approach to China has three principal objectives.

First, we will secure our national security interests and those of our allies and partners, and we will protect human rights. We will clearly communicate to the PRC our concerns about its behavior. And we will not hesitate to defend our vital interests. Even as our targeted actions may have economic impacts, they are motivated solely by our concerns about our security and values. Our goal is not to use these tools to gain competitive economic advantage.

Second, we seek a healthy economic relationship with China: one that fosters growth and innovation in both countries. A growing China that plays by international rules is good for the United States and the world. Both countries can benefit from healthy competition in the economic sphere. But healthy economic competition – where both sides benefit – is only sustainable if that competition is fair. We will continue to partner with our allies to respond to China’s unfair economic practices. And we will continue to make critical investments at home – while engaging with the world to advance our vision for an open, fair, and rules-based global economic order.

Third, we seek cooperation on the urgent global challenges of our day. Since last year’s meeting between Presidents Biden and Xi, both countries have agreed to enhance communication around the macroeconomy and cooperation on issues like climate and debt distress. But more needs to be done. We call on China to follow through on its promise to work with us on these issues – not as a favor to us, but out of our joint duty and obligation to the world. Tackling these issues together will also advance the national interests of both of our countries.

STATE OF OUR ECONOMIES
Let me begin by discussing the state of our economies.

In recent years, many have seen conflict between the United States and China as increasingly inevitable. This was driven by fears, shared by some Americans, that the United States was in decline. And that China would imminently leapfrog us as the world’s top economic power – leading to a clash between nations.

It’s important to know this: pronouncements of U.S. decline have been around for decades. But they have always been proven wrong. The United States has repeatedly demonstrated its ability to adapt and reinvent to face new challenges. This time will be no different – and the economic statistics show why.

Since the end of the Cold War, the American economy has grown faster than most other advanced economies. And over the past two years, we have mounted the strongest post-pandemic recovery among major advanced economies. Our unemployment rate is near historic lows. Real GDP per capita has reached an all-time high, and we have experienced the strongest two-year growth in new businesses on record.

This recovery is made possible by the strength of our economic fundamentals. Of course, this does not mean that our work is finished. Our top economic priority is to rein in inflation while protecting the economic gains of our recovery. A few weeks ago, the United States took decisive action to strengthen public confidence in the banking system after the failures of two regional institutions. The U.S. banking system remains sound, and we will take any necessary steps to ensure the United States continues to have the strongest and safest financial system in the world.

Over the past few decades, China has experienced an impressive economic rise. Between 1980 and 2010, China’s economy grew by an average of 10 percent per year. This led to a truly remarkable feat: the rise of hundreds of millions of people out of poverty. China’s rapid catch-up growth was fueled by its opening-up to global trade and pursuit of market reforms. 

But like many countries, China today faces its share of near-term headwinds. This includes vulnerabilities in its property sector, high youth unemployment, and weak household consumption. In the longer term, China faces structural challenges. Its population is aging, and its workforce is already declining. And it has experienced a sharp reduction in productivity growth – amid its turn toward economic nationalism and policies that substantially increase the government’s intervention in the economy. None of these recent developments detract from China’s progress or the hard work and talent of the Chinese people. But China’s long-run growth rate seems likely to decline. 

Of course, an economy’s size is not the sole determinant of its strength. America is the largest economy in the world, but it also remains an unparalleled leader on a broad set of economic metrics – from wealth to technological innovation. U.S. GDP per capita is among the highest in the world and over five times as large as China’s. More than resources or geography, our country’s success can be attributed to our people, values, and institutions. American democracy, while not perfect, protects the free exchange of ideas and rule of law that is at the bedrock of sustainable growth. Our educational and scientific institutions lead the world. Our innovative culture is enriched by new immigrants, including those from China – enabling us to continue to generate world-class, cutting-edge products and industries.

Importantly, our economic power is amplified because we don’t stand alone. America values our close friends and partners in every region of the world, including the Indo-Pacific. In the 21st century, no country in isolation can create a strong and sustainable economy for its people. That’s why, under President Biden’s leadership, we’ve sought to rebuild and reinvest in our relationships with other countries.

All this to say: China’s economic growth need not be incompatible with U.S. economic leadership. The United States remains the most dynamic and prosperous economy in the world. We have no reason to fear healthy economic competition with any country.

SECURING OUR NATIONAL SECURITY INTERESTS AND PROTECTING HUMAN RIGHTS
There are many challenges before us. But the President and I believe that China and the United States can manage our economic relationship responsibly. We can work toward a future in which both countries share in and drive global economic progress. Whether we can reach this vision depends in large part on what both countries do in the next few years.

Let me speak to our first objective: securing our national security and protecting human rights. These are areas where we will not compromise.

National Security
As in all of our foreign relations, national security is of paramount importance in our relationship with China. For example, we have made clear that safeguarding certain technologies from the PRC’s military and security apparatus is of vital national interest.

We have a broad suite of tools to achieve this aim. When necessary, we will take narrowly targeted actions. The U.S. government’s actions can come in the form of export controls. They can include additions to an entity list that restricts access by those that provide support to the People’s Liberation Army. The Treasury Department has sanctions authorities to address threats related to cybersecurity and China’s military-civil fusion. We also carefully review foreign investments in the United States for national security risks and take necessary actions to address any such risks. And we are considering a program to restrict certain U.S. outbound investments in specific sensitive technologies with significant national security implications.

As we take these actions, let me be clear: these national security actions are not designed for us to gain a competitive economic advantage, or stifle China’s economic and technological modernization. Even though these policies may have economic impacts, they are driven by straightforward national security considerations. We will not compromise on these concerns, even when they force trade-offs with our economic interests.

There are key principles that guide our national security actions in the economic sphere.

First, these actions will be narrowly scoped and targeted to clear objectives. They will be calibrated to mitigate spillovers into other areas. Second, it is vital that these tools are easily understood and enforceable. And they must be readily adaptable when circumstances change. Third, when possible, we will engage and coordinate with our allies and partners in the design and execution of our policies.

In addition, communication is essential to mitigating the risk of misunderstanding and unintended escalation. When we take national security actions, we will continue to outline our policy reasoning to other countries. We will listen and address concerns about unintended consequences.

Among our most pressing national security concerns is Russia’s illegal and unprovoked war against Ukraine. In my visit to Kyiv, I saw firsthand the brutality of Russia’s invasion. The Kremlin has bombed hospitals; destroyed cultural sites; attacked energy grids to cause widespread pain and suffering among civilians. Ending Russia’s war is a moral imperative. It will save many innocent lives. As I’ve said, it is also the single best thing we can do for the global economy. To help end Russia’s war, we have mounted the swiftest, most unified, and most ambitious multilateral sanctions regime in modern history. Our broad coalition of partners has also provided assistance to Ukraine so it can defend itself.

China’s “no limits” partnership and support for Russia is a worrisome indication that it is not serious about ending the war. It is essential that China and other countries do not provide Russia with material support or assistance with sanctions evasion. We will continue to make the position of the United States extremely clear to Beijing and companies in its jurisdiction. The consequences of any violations would be severe.

Human Rights
Like national security, we will not compromise on the protection of human rights. This principle is foundational to how we engage with the world.

With our own eyes, the world has seen the PRC government escalate its repression at home. It has deployed technology to surveil and control the Chinese people – technology that it is now exporting to dozens of countries.

Human rights abuses violate the world’s moral conscience. They also violate the foundational principles of the United Nations – which virtually every country, including China, has signed onto. The United States will continue to use our tools to disrupt and deter human rights abuses wherever they occur around the globe.

In public and in private with Beijing, the United States has raised serious concerns about the PRC government’s abuses in Xinjiang, as well as in Hong Kong, Tibet, and other parts of China. And we have and will continue to take action. We have imposed sanctions on the PRC’s regional officials and companies for a range of human rights abuses – from torture to arbitrary detention. And we are restricting imports of goods produced with forced labor in Xinjiang.

Across these actions, we are working in concert with our allies – knowing that we are more effective when we all go at it together. 

III. TOWARDS HEALTHY ECONOMIC ENGAGEMENT
As we protect our security interests and human rights values, we will also pursue our second objective: healthy economic engagement that benefits both countries.

Let’s start with the obvious. The U.S. and China are the two largest economies in the world. And we are deeply integrated with one another. Overall trade between our countries reached over $700 billion in 2021. We trade more with China than with any countries other than Canada and Mexico. American firms have extensive operations in China. Hundreds of Chinese firms are listed on our stock exchanges, which are part of the deepest and most liquid capital markets in the world. According to the Nature Index, the United States and China are each other’s most significant scientific collaborators. And China remains among the top sources for international students in the United States.

As I’ve said, the United States will assert ourselves when our vital interests are at stake. But we do not seek to “decouple” our economy from China’s. A full separation of our economies would be disastrous for both countries. It would be destabilizing for the rest of the world. Rather, we know that the health of the Chinese and U.S. economies is closely linked. A growing China that plays by the rules can be beneficial for the United States. For instance, it can mean rising demand for U.S. products and services and more dynamic U.S. industries.

Modern Supply-Side Investments at Home
In April 2021, I delivered my first major international economic policy speech as Treasury Secretary. I said that “credibility abroad begins with credibility at home.” At a basic level, America’s ability to compete in the 21st century turns on the choices that Washington makes – not those that Beijing makes.

Our economic strategy is centered around investing in ourselves – not suppressing or containing any other economy.

In the two years since my speech, the United States has pursued an economic agenda that I call modern supply-side economics. Our policies are designed to expand the productive capacity of the American economy. That is, to raise the ceiling for what our economy can produce. To do so, President Biden has signed three historic bills into law. We’ve enacted the Bipartisan Infrastructure Law – our generation’s most ambitious effort to modernize roads, bridges, and ports and broaden access to high-speed Internet. We’ve mounted a historic expansion of American semiconductor manufacturing through the CHIPS and Science Act. And we are making our nation’s largest investment in clean energy with the Inflation Reduction Act. These actions have fortified U.S. strength in the industries of the future. And they are lifting our long-term economic outlook.

Our Vision and Conditions for Healthy Economic Competition
It’s important to understand the nature of the healthy economic competition that the United States is pursuing.

The United States does not seek competition that is winner-take-all. Instead, we believe that healthy economic competition with a fair set of rules can benefit both countries over time. A basic principle of economics is that sustained, repeated competition can lead to mutual improvement. Sports teams perform at a higher level when they consistently face top rivals. Firms produce better and cheaper goods when they compete for consumers. There is a world in which, as companies in the U.S. and China challenge each other, our economies can grow, standards of living can rise, and new innovations can bear fruit.

For example, China has benefited from American inventions like the personal computer and the MRI. In the same way, I believe that new scientific and medical developments from China can benefit Americans and the world – and spur us to undertake even more leading-edge research and innovation.

But this type of healthy competition is only sustainable if it is fair to both sides.

China has long used government support to help its firms gain market share at the expense of foreign competitors. But in recent years, its industrial policy has become more ambitious and complex. China has expanded support for its state-owned enterprises and domestic private firms to dominate foreign competitors. It has done so in traditional industrial sectors as well as emerging technologies. This strategy has been coupled with aggressive efforts to acquire new technological know-how and intellectual property – including through IP theft and other illicit means.

Government intervention can be justified in certain circumstances – such as to correct specific market failures. But China’s government employs non-market tools at a much larger scale and breadth than other major economies. China also imposes numerous barriers to market access for American firms that do not exist for Chinese businesses in the United States. For example, Beijing has often required foreign firms to transfer proprietary technology to domestic ones – simply to do business in China. These limits on access to the Chinese market tilt the playing field in favor of Chinese firms. Further, we are concerned about a recent uptick in coercive actions targeting U.S. firms, which comes at the same moment that China states that it is re-opening for foreign investment.

The actions of China’s government have had dramatic implications for the location of global manufacturing activity. And they have harmed workers and firms in the U.S. and around the world.

In certain cases, China has also exploited its economic power to retaliate against and coerce vulnerable trading partners. For example, it has used boycotts of specific goods as punishment in response to diplomatic actions by other countries. China’s pretext for these actions is often commercial. But its real goal is to impose consequences on choices that it dislikes – and to force sovereign governments to capitulate to its political demands.

The irony is that the open, fair, and rules-based global economy that the United States is calling for is the very same international order that helped make China’s economic transformation possible. And the inefficiencies and vulnerabilities generated by China’s unfair practices may end up hurting its own growth.

China’s senior officials have repeatedly spoken about the importance of allowing markets to play a “decisive role” in resource allocation – including in a speech just earlier this year. It would be better for China and the world if Beijing were to actually shift policies in these directions and meet its own stated reform ambitions.

As we press China on its unfair economic practices, we will continue to take coordinated actions with our allies and partners in response. A top priority for President Biden is the resilience of our critical supply chains. In certain sectors, China’s unfair economic practices have resulted in the over-concentration of the production of critical goods inside China. Under President Biden’s leadership, we are not only investing in manufacturing at home. We are also pursuing a strategy called “friendshoring” that is aimed at mitigating vulnerabilities that can lead to supply disruptions. We are creating redundancies in our critical supply chains with the large number of trading partners that we can count on.

Of course, we know that the best way for us to strengthen the global economic order is to show the world that it works. Our investments in the international financial institutions and efforts to deepen our ties around the world are enabling more people to benefit from the international economic system. We are also accelerating our commitments in the developing world. For example, the United States and the rest of the G7 aim to mobilize $600 billion in high-quality infrastructure investments by 2027. Our focus is on projects that generate positive economic returns and foster sustainable debt for these countries. And when the international system needs updating, we will not hesitate to do so. The United States is working with shareholders to evolve the multilateral development banks to better combat today’s pressing global challenges – like climate change, pandemics, and fragility and conflict.

LEADING TOGETHER ON GLOBAL CHALLENGES
As we set the terms of our economic engagement with China, we will also pursue our third objective: cooperation on major global challenges. It is important that we make progress on global issues regardless of our other disagreements. That’s what the world needs from its two largest economies.

As a foundation, we must continue to develop steady lines of communication between our countries for macroeconomic and financial cooperation. Economic developments in the United States and China can quickly ripple through global financial markets and the broader economy. We must maintain a robust exchange of views about how we are responding to economic shocks. My conversations with Vice Premier Liu He and China’s other senior officials have been a good start. I hope to build on them with my new counterpart.

Beyond the macroeconomy, there are two specific global priorities I’d like to highlight today: debt overhang and climate change. These issues can best be managed if both countries work together, and in concert with our allies and partners.

Debt Overhang
First, we must work together to help emerging markets and developing countries facing debt distress. The issue of global debt is not a bilateral issue between China and the United States. It is about responsible global leadership. China’s status as the world’s largest official bilateral creditor imposes on it the same inescapable set of responsibilities as those on other official bilateral creditors when debt cannot be fully repaid.

China’s participation is essential to meaningful debt relief. But for too long, it has not moved in a comprehensive and timely manner. It has served as a roadblock to necessary action.

Earlier this year, I felt the urgency of debt relief firsthand during my visit to Zambia. Government and business leaders spoke to me about how Zambia’s debt overhang has held back critical public and private investment and depressed economic development. But Zambia is not the only country in this situation. The IMF estimates that more than half of low-income countries are close to or already in debt distress. 

The United States has had extensive discussions with Beijing about the need for speedy debt treatment. We welcome China’s recent provision of specific and credible financing assurances for Sri Lanka, which has enabled the IMF to move forward with a program. But now, all of Sri Lanka’s bilateral creditors – including China – will need to deliver debt treatments in line with their assurances in a timely manner. We continue to urge China’s full participation to provide debt treatments in other cases in line with IMF parameters. This includes urgent cases like Zambia and Ghana.

Prompt action on debt is in China’s interest. Delaying needed debt treatments raises the costs both for borrowers and creditors. It worsens borrowers’ economic fundamentals and increases the amount of debt relief they will eventually need.

More broadly, there is considerable room for improvement in the international debt restructuring process. With the IMF and World Bank, we are working with a range of stakeholders to improve the Common Framework process for low-income countries and the debt treatment process more generally. As I heard from Zambian officials, solving these issues is a true test of multilateralism.

Climate Change
Second, we must work together to tackle longstanding global challenges that threaten us all. Climate change is at the top of that list. History shows us what our two countries can do: moments of climate cooperation between the United States and China have made global breakthroughs possible, including the Paris Agreement.

We have a joint responsibility to lead the way. China is the largest emitter of greenhouse gases, followed by the United States. The U.S. will do its part. Over the past year, the United States has taken the boldest domestic climate action in our nation’s history. Our investments put us on track to meet U.S. commitments under the Paris Agreement and achieve net-zero by 2050. And they will have positive spillovers for the world, including through reductions in the costs of clean energy technologies. We are also working abroad to help countries make a just energy transition to reduce their carbon emissions. These transitions will also help expand energy access and provide economic opportunity for impacted communities and workers.

We expect China to deliver on its commitments in our Joint Glasgow Declaration. This includes meeting mitigation targets and ending overseas financing of unabated coal-fired power plants. China should also support developing countries and emerging markets in their clean energy transitions. Further, we look forward to working together to boost private capital flows as co-chairs of the G20 working group on sustainable finance.

We stand ready to work with China on the existential challenge of climate change. And we urge China to seriously engage with us and deliver on its commitments. The stakes are too high not to.

CONCLUSION
Some see the relationship between the U.S. and China through the frame of great power conflict: a zero-sum, bilateral contest where one must fall for the other to rise.

President Biden and I don’t see it that way. We believe that the world is big enough for both of us. China and the United States can and need to find a way to live together and share in global prosperity. We can acknowledge our differences, defend our own interests, and compete fairly. Indeed, the United States will continue to proceed with confidence about the fundamental strength of the American economy and the skill of American workers. But as President Biden said, “we share a responsibility…to prevent competition from becoming anything ever near conflict.”

Negotiating the contours of engagement between great powers is difficult. And the United States will never compromise on our security or principles. But we can find a way forward if China is also willing to play its part.

That’s why I plan to travel to China at the appropriate time. My hope is to engage in an important and substantive dialogue on economic issues with my new Chinese government counterpart following the political transition in Beijing. I believe this dialogue can help lay the groundwork for responsibly managing our bilateral relationship and cooperating on areas of shared challenge to our nations and the world.

As you know, I am an economist by trade. Economics is popularly seen as a field concerning the structure and performance of entire economies. But at its most granular level, economics is much more foundational. It’s the study of the choices that people make. Specifically, how people make choices under specific circumstances – of scarcity, of risk, and sometimes, of stress. And how choices by individuals and firms affect one another, and how they add up to a national or global picture.

In other words, an economy is just an aggregate of choices that people make.

The relationship between the United States and China is the same. Our path is not preordained, and it is not destined to be costly. The trajectory of this relationship is the aggregate of choices that all of us in these two great powers make over time – including when to cooperate, when to compete, and when to recognize that even amid our competition, we have a shared interest in peace and prosperity.

The United States believes that responsible economic relations between the U.S. and China is in the self-interest of our peoples. It is the hope and expectation of the world. And at this moment of challenge, I believe it must be the choice that both countries – the United States and China – make.

Thank you.

"""

tokens = tokenizer.encode_plus(txt, add_special_tokens=False,
                               return_tensors='pt')

print(len(tokens['input_ids'][0]))
print(tokens)

a = torch.arange(10)
print(a)

torch.split(a, 4)

input_id_chunks = tokens['input_ids'][0].split(510)
mask_chunks = tokens['attention_mask'][0].split(510)

a = torch.cat(
    [torch.Tensor([101]), a, torch.Tensor([102])]
)

print(a)

padding_len = 20 - a.shape[0]

print(padding_len)

if padding_len > 0:
    a = torch.cat(
        [a, torch.Tensor([0] * padding_len)]
    )

print(a)

# define target chunksize
chunksize = 512

# split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))

# loop through each chunk
for i in range(len(input_id_chunks)):
    # add CLS and SEP tokens to input IDs
    input_id_chunks[i] = torch.cat([
        torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
    ])
    # add attention tokens to attention mask
    mask_chunks[i] = torch.cat([
        torch.tensor([1]), mask_chunks[i], torch.tensor([1])
    ])
    # get required padding length
    pad_len = chunksize - input_id_chunks[i].shape[0]
    # check if tensor length satisfies required chunk size
    if pad_len > 0:
        # if padding length is more than 0, we must add padding
        input_id_chunks[i] = torch.cat([
            input_id_chunks[i], torch.Tensor([0] * pad_len)
        ])
        mask_chunks[i] = torch.cat([
            mask_chunks[i], torch.Tensor([0] * pad_len)
        ])

# check length of each tensor
for chunk in input_id_chunks:
    print(len(chunk))
# print final chunk so we can see 101, 102, and 0 (PAD) tokens are all correctly placed
print(chunk)

input_ids = torch.stack(input_id_chunks)
attention_mask = torch.stack(mask_chunks)

input_dict = {
    'input_ids': input_ids.long(),
    'attention_mask': attention_mask.int()
}
print(input_dict)

outputs = model(**input_dict)
probs = torch.nn.functional.softmax(outputs[0], dim=-1)
probs = probs.mean(dim=0)
print(probs)

pred=torch.argmax(probs)

if pred.item() == 0:
    print("good")
if pred.item() == 1:
    print("bad")
if pred.item() == 2:
    print("neutral")