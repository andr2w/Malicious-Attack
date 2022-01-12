from data import read_data


features, labels = read_data('fake')

print(len(features))
print(len(labels))
print(features[0])