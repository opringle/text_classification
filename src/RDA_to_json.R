#load in Rda file
print("loading data...")
load("../data/tp_sample_data_v0.Rda")

#convert to json
print("converting to json object...")
library(rjson)
x <- toJSON(unname(split(data_v0, 1:nrow(data_v0))))
cat(x)

#export to file
print("exporting file...")
fileConn<-file("./data/transit_data.json")
writeLines(x, fileConn)
close(fileConn)

print("done :)")
