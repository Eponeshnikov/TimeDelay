list_time = []
etalist: float = 0


def PrintProgressBar(_time, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    q = int(0.001 * total)
    if q == 0:
        q = 1
    d = q * 4
    try:
        global list_time
        global etalist
        list_time.append(_time)
        if len(list_time) == d:
            etalist = sum(list_time) / (len(list_time))
            list_time[:q] = []
        eta = round((total - iteration) * etalist)
        etah = eta // 3600
        etamin = eta // 60 - etah * 60
        etasec = eta - etamin * 60 - etah * 3600
        sec: str = 'ETA: {0} h {1} min {2} sec   '.format(str(etah), str(etamin), str(etasec))
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        number = '({0}/{1})'.format(str(iteration), str(total))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s %s %s' % (prefix, bar, percent, number, suffix, sec), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
    except Exception:
        print(Exception.__name__)
