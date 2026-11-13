# Data Report (Member 1)

**Dataset:** Spam Email Dataset (Kaggle, mail_data.csv). Short SMS-style messages labelled ham/spam.
**Total messages (after cleaning):** 5,147
**Spam:** 12.5%   **Not spam:** 87.5%
**Cleaning done:** removed empty rows and duplicates, converted labels (spam = 1, ham = 0), renamed columns to `text` and `label`.
**Average length:** spam 23.7 words, not spam 14.3 words.

## Findings
- The data is imbalanced (only 12.5% spam), so models should use class weights and be judged by precision, recall and F1.
- Spam messages are longer on average, so message length can be a feature.
- The text contains encoding noise (for example `Â£` and `&lt;#&gt;`), which preprocessing should remove.
- The data is short messages, not full emails. A real email dataset (Enron or SpamAssassin) could be added later to improve coverage.

## Files
- emails_raw.csv: full cleaned dataset (columns: text, label; 1 = spam)
- sample_100.csv: small file for quick testing
- prepare_data.py: creates the files above from mail_data.csv
- eda.py: creates the three charts
- chart_*.png: EDA charts