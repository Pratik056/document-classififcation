import os
import fitz
import pandas as pd

def get_path():
    final_path = []
    path1= input('Enter AI path files')
    print("Path register successfully")
    path2= input('Enter WEB path files')
    print("Path register successfully")
    final_path.append(path1)
    final_path.append(path2)
    return final_path


def get_final_dataframe(path, flag):
    df=pd.DataFrame(columns=['Text','Label'])
    content = []
    label = []
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            doc = fitz.open(path + '\\' + file)
            content_temp = ""
            for page in range(len(doc)):
                content_temp = content_temp + doc[page].get_text()
            content.append(content_temp)
    df['Text']= content
    df['Label']= flag
    return df

def get_content_of_pdfs(file_path):
    for path in file_path:
        if '\\AI' in path:
            print("asbkasbkj")
            df_ai = get_final_dataframe(path,1)
        elif '\\WEB' in path:
            df_web = get_final_dataframe(path,0)
    df = df_ai.append(df_web)
    print(df)
    return df 
                      

def get_content(file_path):
    df = pd.DataFrame(columns=['Text', 'Label'])
    df = get_content_of_pdfs(file_path)
    return df
    

def dataset_generate():
    file_path = get_path()
    dataset = get_content(file_path)
    dataset.to_csv("dataset.csv")
    
    # print(file_path)


if __name__ == '__main__':
    dataset_generate()