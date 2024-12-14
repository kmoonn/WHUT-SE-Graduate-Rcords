import os

courses = {
    'Advanced Artificial Intelligence Principles and Technologies':{
        'name':'高级人工智能原理与技术','teacher':'彭德巍','score':'-1'
    },
    'Algorithm Analysis and Theory' : {
        'name':'算法分析与原理', 'teacher':'高曙', 'score':'-1'
    },
    'Networks Groups and Markets': {
        'name':'网络、群体与市场', 'teacher':'石兵', 'score':'-1'
    },
    'Modern Software Engineering': {
        'name':'现代软件工程学','teacher':'邱奇志','score':'-1'
    },
    'Scientific and Technical English Training': {
        'name':'科技英语实训','teacher':'邱奇志','score':'-1'
    },
    'Sports Nutrition': {
        'name':'运动营养学','teacher':'仲鹏飞','score':'-1'
    },
    'Marriage, Workplace, Personality': {
        'name':'婚恋·职场·人格','teacher':'魏超','score':'-1'
    },
    'Postgraduate Students’ Stress and Emotions': {
        'name':'研究生的压力与情绪','teacher':'张琴','score':'-1'
    },
    'Information Security Technology': {
        'name':'信息安全技术','teacher':'孟伟','score':'-1'
    },
    'Statistical Computing': {
        'name':'统计计算','teacher':'王传美','score':'-1'
    },
    'High Performance Computer Networks': {
        'name':'高性能计算机网络','teacher':'颜昕','score':'-1'
    },
    }

def make_dirs():
    for course in courses:
        if not os.path.exists(course):
            os.mkdir(course)

def show_contents():
    files = os.listdir()
    for file in files:
        if file in courses.keys():
            print(f'{file} {" ".join(courses[file].values())}')
        else:
            continue

if __name__ == '__main__':
    make_dirs()
    # show_contents()