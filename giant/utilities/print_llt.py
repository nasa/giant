from typing import Sequence

from numpy.typing import NDArray

def print_llt(labels: Sequence[str], values: NDArray, min_column_length: int = 10):
    """
    This pretty prints the lower left triangle of a matrix with labels.

    This is used to print covariance and correlation matrices.

    :param labels: The labels for each row/column of the matrix
    :param values: The matrix to print
    :param min_column_length: the minimum length of a column in characters
    """

    # get the maximum length of the labels, with a minimum size 
    max_label = max(max(map(len, labels)), min_column_length)

    # make the label format string based on the maximum label length
    label_format = '{:<' + str(max_label) + 's}'

    # make the value format string based on the maximum label length
    value_format = '{:>' + str(max_label) + '.' + str(max_label-7) + 'e}'

    # loop through the rows
    for rind, rlabel in enumerate(labels):
        # print the label format at the beginning of each new row.  Don't use a new line after
        print(label_format.format(rlabel), end='  ')
        # loop through the columns
        for cind, clabel in enumerate(labels):
            # skip the upper right triangle
            if cind > rind:
                print('\n', end='')
                break
            # print out the value using the format string.  Don't use a new line after
            print(value_format.format(values[rind, cind]), end='  ')

    # print out a space to get the column labels in the right place
    print('\n' + label_format.format(''), end='')

    # change the label format to be right aligned
    label_format = label_format.replace('<', '>')

    # print out a row of column labels
    for clabel in labels:

        print(label_format.format(clabel), end='  ')

    # print a new line
    print('')

    return max_label

