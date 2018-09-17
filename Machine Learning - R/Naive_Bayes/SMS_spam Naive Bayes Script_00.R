wd <- getwd()

if (wd != "C:/Users/pierr/Documents/GitHub/DS_Portfolio/Machine Learning - R/Naive_Bayes") {
  setwd("C:/Users/pierr/Documents/GitHub/DS_Portfolio/Machine Learning - R/Naive_Bayes/")
} else {
  print("Already set!")
}

if("SMSSpamCollection.txt" %in% list.files()) {
  sms_raw <- read.table("SMSSpamCollection.txt", header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
} else {
  download.file("http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/smsspamcollection.zip",
                destfile = "SMS_spam.zip")
  
  unzip("SMS_spam.zip")
  sms_raw <- read.table("SMSSpamCollection.txt", header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE)
}

colnames(sms_raw) <- c("type", "text")
sms_raw$type <- factor(sms_raw$type)

# Shuffle the dataset
set.seed(123)
sms_raw <- sms_raw[sample(nrow(sms_raw)), ]
str(sms_raw)
