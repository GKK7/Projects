# import the required module
import wikipedia

#select the topic first to get a summary
topic=input("Select the search topic \n>")
print(wikipedia.search(topic))

subtopic=input("Select the subtopic from the options \n>")
print(wikipedia.summary(subtopic),end="")
