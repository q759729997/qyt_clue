if __name__ == "__main__":
    """平均分计算"""
    text = '0.428571；0.43685；0.437326'
    scores = text.split('；')
    scores = [float(temp) for temp in scores]
    print(scores)
    avg_score = sum(scores)/len(scores)
    print(avg_score)
    print(round(avg_score, 4))
