def processText(series, case='normal', remove_chars=[]):
    '''
    Process text data

    :param series: Text data list
    :param case:
        normal : does not change any
        upper  : convert to upper case
        lower  : convert to lower case

    :param remove_chars:
        remove characters from texts
    :return:
        processed text list
    '''
    series = map(str,list(series))
    for i in range(len(series)):
        if case == 'upper':
            series[i] = series[i].upper()
        elif case == 'lower':
            series[i] = series[i].lower()

        for char in remove_chars:
            series[i] = series[i].replace(char, '')
    return series

