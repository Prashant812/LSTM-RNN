# Classification of Handwritten Character Dataset and Consonant Vowel (CV) segment dataset
## Objective
The main aim of this project is to classify Handwritten character dataset which consist of Kannada/Telugu script in coordinate form and
to also classify Consonant Vowel (CV) segment dataset, a conversational speech data spoken in Hindi language by using
RNN and LSTM.

## Handwritten Character Dataset
### Data

Five characters are there a, aI, bA, dA and lA, each characters are stored in .txt files as sequence of 2-dimensional points (x and y
coordinates) :-
</br>
<p align="center">
  <img width="450" src="https://github.com/Prashant812/Sale-Tracker/assets/93676625/75046862-a09a-4860-b808-7a6418089b02" >
</p>

### Model
1. RNN
* Accuracy on Train Set: 0.96
* Accuracy on Test Set: 0.94
2. LSTM
* Accuracy on Train Set: 0.98
* Accuracy on Test Set: 0.97

### Confusion Matrix
<p align="center">
  <img width="300" src="https://github.com/Prashant812/Sale-Tracker/assets/93676625/1bef7651-0e9d-4bc0-9cc5-dd85cd21fc05" >
  <img width="250" src="https://github.com/Prashant812/Sale-Tracker/assets/93676625/39f4a5fb-8236-43a2-8149-9b9ebaeef4d5" >
</p>


## Consonant Vowel (CV) segment dataset

### Data
This dataset consists of subset of CV segments from a conversational speech data spoken in Hindi language. Training and test data are separated and are 
provided inside the respective CV segment folder where each class consist of 39-dimensional Mel frequency cepstral 
coefficient (MFCC) features.


### Model
1. RNN
* Accuracy on Train Set: 0.988
* Accuracy on Test Set: 0.899
2. LSTM
* Accuracy on Train Set: 0.997
* Accuracy on Test Set: 0.879

### Confusion Matrix
<p align="center">
  <img width="300" src="https://github.com/Prashant812/Sale-Tracker/assets/93676625/e60a72c1-d595-44fc-a555-bf5276415b61" >
</p>

## Conclusion
In both of the cases, our data consists of long sequential sequences, the better accuracy of the LSTM model confirms its effectiveness over the standard RNN. 
