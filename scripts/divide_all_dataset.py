import random
import os

def split_dataset(dataset, train_ratio, val_ratio):
    random.shuffle(dataset)
    total_samples = len(dataset)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size+val_size]
    test_set = dataset[train_size+val_size:]
    
    return train_set, test_set, val_set

def check_file_existence(file_path):
    return os.path.isfile(file_path)

def save_dataset(filename, dataset):
    with open(filename, 'w') as file:
        for data in dataset:
            file.write(data + '\n')

def count_valid_invalid_data(dataset):
    valid_count = 0
    invalid_count = 0
    total = 0
    
    for data in dataset:
        total += 1

        image_path, label_path = data.split("  ")
        
        if check_file_existence(image_path) and check_file_existence(label_path):
            valid_count += 1
        else:
            invalid_count += 1
    
    return valid_count, invalid_count, total

def count_files(directory):
    count = 0
    
    for root, dirs, files in os.walk(directory):
        count += len(files)
    
    return count

def main():
    dataset = []
    
    # 读取数据集，每行作为一个数据，将其存储在列表中
    with open('all_lane_dataset.txt', 'r') as file:
        for line in file:
            data = line.strip()
            image_path, label_path = data.split("  ")
            
            if check_file_existence(image_path) and check_file_existence(label_path):
                dataset.append(data)
            else:
                print("文件不存在:", image_path, label_path)
    
    train_ratio = float(input("训练集比例："))
    val_ratio = float(input("验证集比例："))
    
    test_ratio = 1 - train_ratio - val_ratio
    
    train_set, test_set, val_set = split_dataset(dataset, train_ratio, val_ratio)
    
    save_dataset('all_train_data.txt', train_set)
    save_dataset('all_test_data.txt', test_set)
    save_dataset('all_val_data.txt', val_set)
    
    valid_count, invalid_count, total = count_valid_invalid_data(dataset)
    
    print("有效数据个数:", valid_count)
    print("无效数据个数:", invalid_count)
    print("文件总数:", total)
    print("训练集总数:", len(train_set))
    print("验证集总数:", len(val_set))
    print("测试集总数:", len(test_set))



if __name__ == "__main__":
    main()