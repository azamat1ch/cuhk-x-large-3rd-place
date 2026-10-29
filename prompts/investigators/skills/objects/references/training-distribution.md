# Complete training object distribution

Source: the released-training partition of the object options. This reference
contains only aggregate training labels, with no evaluation labels, recordings or identities.
Scope: all 133 released training object-interaction questions; 30 offered names.
Normalization lowercases wording and removes the leading article. Phone and
smartphone remain separate strings. Counts overlap because each question offers
multiple names. Correct/offered is descriptive, not held-out predictive accuracy.

| Offered object | Correct / offered | Supplied correct labels when this object is offered |
| --- | ---: | --- |
| blanket | 0/20 | clothes 20 |
| book | 0/17 | documents 17 |
| bottle | 0/10 | glass 10 |
| cabinet | 10/10 | cabinet 10 |
| calculator | 0/43 | keyboard 9, phone 12, remote 9, smartphone 13 |
| cloth | 10/30 | cloth 10, clothes 20 |
| clothes | 20/35 | cabinet 5, cloth 10, clothes 20 |
| cup | 0/10 | glass 10 |
| documents | 20/20 | documents 20 |
| drawer | 0/9 | cabinet 9 |
| fridge | 0/8 | cabinet 8 |
| glass | 10/10 | glass 10 |
| headphones | 7/7 | headphones 7 |
| headset | 0/5 | headphones 5 |
| keyboard | 10/26 | keyboard 10, phone 9, remote 7 |
| magazine | 0/15 | documents 15 |
| mouse | 0/5 | keyboard 5 |
| mug | 0/10 | glass 10 |
| napkin | 0/3 | cloth 3 |
| newspaper | 0/13 | documents 13 |
| notebook | 0/15 | documents 15 |
| phone | 18/35 | headphones 3, keyboard 7, phone 18, remote 7 |
| remote | 10/39 | keyboard 4, phone 10, remote 10, smartphone 15 |
| shelf | 0/8 | cabinet 8 |
| smartphone | 18/18 | smartphone 18 |
| speaker | 0/7 | headphones 7 |
| sponge | 0/8 | cloth 8 |
| tablet | 0/33 | keyboard 5, phone 12, smartphone 16 |
| towel | 0/35 | cloth 9, clothes 20, headphones 6 |
| wallet | 0/28 | phone 11, remote 7, smartphone 10 |

The supplied correct labels use 10 strings, or nine families when phone and
smartphone are grouped. Neither of those two strings co-occurs with the other
in these object questions. Removing all never-correct names would leave one
candidate in only 44/133 questions; the other 89 retain two or three. This is
an in-sample diagnostic, not an elimination policy or measured model improvement.

Conditional counts prevent overgeneralization: towel can distract from clothes,
cloth or headphones; calculator and remote can appear with several real device
classes. Cloth can distract from clothes, and clothes can distract from cloth or
cabinet. A training-perfect option is still only a naming prior for a compatible
interaction, never evidence that its action occurred.

Among these 133 training object questions, 38 also have a single-action question.
Their supplied labels pair bowl wiping with cloth; garment folding/dressing with
clothes; typing with keyboard; calling/phone use with phone; reading/page turning
with documents; headphone placement with headphones; remote use with remote;
phone use/selfie with smartphone; cabinet opening with cabinet; drinking with
glass. These are label co-occurrences, not independent visual verification or
universal action-to-object rules. The other 95 lack that paired action question.
