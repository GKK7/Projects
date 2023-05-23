from datasets import load_dataset, Image, Dataset

dataset = Dataset.from_dict({"image": ["/home/gkirilov/Jenna_Ortega_resized/0.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/1.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/2.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/3.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/4.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/5.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/6.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/7.jpg",
                                       "/home/gkirilov/Jenna_Ortega_resized/8.jpg"]}).cast_column("image", Image())
print(dataset[0]["image"])