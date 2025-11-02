"""
File for reading CSV files and returning a 2D list
"""
import pandas as pd
import numpy as np
import ast
import pickle

# Custom function to handle strings with space-separated numbers and convert them back to NumPy arrays
def convert_to_array(array_str):
    try:
        # Remove any unwanted characters like square brackets and split by space
        cleaned_str = array_str.replace('[', '').replace(']', '').strip()
        # Split the string by spaces and convert the result to a NumPy array of floats
        return np.array([float(x) for x in cleaned_str.split()])
    except (ValueError, AttributeError):
        # If the string cannot be converted, return it as is (error handling)
        return array_str

def to_array_if_sequence(val):
    if isinstance(val, np.ndarray):
        return val
    elif val is None:
        return None
    elif isinstance(val, (int, float)):
        return np.array([val])
    elif isinstance(val, list):
        return np.array(val)
    elif isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try:
            return np.fromstring(val.strip('[]'), sep=' ')
        except:
            return val  # fallback in case parsing fails
    else:
        return [val]  # wrap scalars


def read_data_df(filename, datatype=None, truedataindex=None, outtype='np.array',return_data_info=True):
    """
    Parameters
    ----------
    filename : str
        Name of the pickled file.

    datatype : list, optional
        List of data types as strings. Default is None.

    truedataindex : list, optional
        List of indices for assimilation. Default is None.

    outtype : str, optional
        Type of output data. Default is 'np.array'.

    Returns
    -------
    flat_array : flat numpy array containing all the data. This is returned if outtype is 'np.array'.
    data : list of dictionaries with keys equal to column names. This is returned if outtype is 'list'.

    If return_data_info is True, the function will also return the data keys and the data info (column names and index).
    """

    # read the file
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, index_col=0)
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    # convert the string representation of arrays back to NumPy arrays
    for col in df.columns:
        df[col] = df[col].apply(convert_to_array)

    df = df.where(pd.notnull(df), None)

    if outtype == 'np.array': # vectorize data
        if datatype is not None:
            if truedataindex is not None:
                flat_array = np.concatenate([np.concatenate([df.iloc[ti][col] if isinstance(df.iloc[ti][col], np.ndarray) else
                                                             np.array([df.iloc[ti][col]])
                                                            for col in datatype]) for ti in truedataindex])
                if return_data_info:
                   return flat_array, list(datatype), [df.index[el] for el in truedataindex]
            else:
                flat_array = np.concatenate([np.concatenate([row[col] if isinstance(row[col], np.ndarray) else
                                                             np.array([row[col]])
                                                for col in datatype]) for _, row in df.iterrows()])
                if return_data_info:
                   return flat_array, list(datatype), list(df.index)
        else:
            if truedataindex is not None:
                flat_array = np.concatenate([np.concatenate([df.iloc[ti][col] if isinstance(df.iloc[ti][col], np.ndarray) else
                                                             np.array([df.iloc[ti][col]])
                                                            for col in df.columns]) for ti in truedataindex])
                if return_data_info:
                    return flat_array, list(df.columns), [df.index[el] for el in truedataindex]
            else:
                flat_array = np.concatenate([np.concatenate([row[col] if isinstance(row[col], np.ndarray) else np.array([row[col]])
                                for col in df.columns]) for _, row in df.iterrows()])
                if return_data_info:
                    return flat_array, list(df.columns), list(df.index)

        return flat_array

    elif outtype == 'list': # return data as a list over row indices. Where each list element is a dictionary with keys equal to column names
        if datatype is not None:
            if truedataindex is not None:
                data = [
                    {
                        col: to_array_if_sequence(df.iloc[ti][col])
                        for col in datatype
                    }
                    for ti in truedataindex
                ]
                
                if return_data_info:
                    data, list(datatype), [df.index[el] for el in truedataindex]
            else:
                data = [
                    {
                        col: to_array_if_sequence(row[col])
                        for col in datatype
                    }
                    for _, row in df.iterrows()
                ]
                if return_data_info:
                    data, list(datatype), list(df.index)
        else:
            if truedataindex is not None:
                data = [
                    {
                        col: to_array_if_sequence(df.iloc[ti][col])
                        for col in df.columns
                    }
                    for ti in truedataindex
                ]
                if return_data_info:
                    data, list(datatype), list(df.index)
            else:
                data = [
                    {
                        col: to_array_if_sequence(row[col])
                        for col in df.columns
                    }
                    for _, row in df.iterrows()
                ]
                if return_data_info:
                    return data, list(df.columns), list(df.index)
        return data

def read_var_df(filename, datatype=None, truedataindex=None, outtype='list'):
    """
    Reads a CSV file and returns a list of dictionaries containing the data.

    Parameters
    ----------
    filename : str
        Name of the CSV file.
    datatype : list, optional
        List of data types as strings. Default is None.
    truedataindex : list, optional
        List of indices for assimilation. Default is None.
    outtype : str, optional
        Type of output data. Default is 'list'.

    Returns
    -------
    var : list
        List of dictionaries with keys equal to column names.
    """

    # read the file
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, index_col=0)
        df.index = df.index.astype(str)  # Convert index to string
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(filename)
    
    # Perform a one-time conversion of datatype if needed
    if datatype is not None:
        try:
            datatype = [ast.literal_eval(col) for col in datatype]
        except (ValueError, SyntaxError):
            pass  # Keep datatype as is if conversion fails


    if outtype == 'list':
        if datatype is not None:
            if truedataindex is not None:
                var = [{col: df.loc[ti][col] for col in datatype} for ti in truedataindex]
            else:
                var = [{col: row[col] for col in datatype} for _, row in df.iterrows()]
        else:
            if truedataindex is not None:
                var = [{col: df.loc[ti][col] for col in df.columns} for ti in truedataindex]
            else:
                var = [{col: row[col] for col in df.columns} for _, row in df.iterrows()]

        return var

def read_data_csv(filename, datatype, truedataindex):
    """
    Parameters
    ----------
    filename:
        Name of csv-file
    datatype:
        List of data types as strings
    truedataindex:
        List of where the "TRUEDATA" has been extracted (e.g., at which time, etc)

    Returns
    -------
    some-type:
        List of observed data
    """

    df = pd.read_csv(filename)  # Read the file

    imported_data = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row corresponding to truedataindex
            row = row.values[0]  # select the values of the dataframe row
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(row[pos[col]]) == str) and (np.isnan(row[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(row[pos[col]])
                    except:  # It is a string
                        csv_data[col] = row[pos[col]]
            imported_data.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of the header in datatype
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            pos = list(range(df.shape[1]))  # Assume the data is in the correct order
            csv_data = [None] * len(temp)
            for col in range(len(temp)):
                if (not type(temp[col]) == str) and (np.isnan(temp[col])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(temp[col])
                    except:  # It is a string
                        csv_data[col] = temp[col]
            imported_data.append(csv_data)

        for rows in df.values:
            csv_data = [None] * dnumber
            for col in range(dnumber):
                if (not type(rows[pos[col]]) == str) and (np.isnan(rows[pos[col]])):  # do not check strings
                    csv_data[col] = 'n/a'
                else:
                    try:  # Making a float
                        csv_data[col] = float(rows[pos[col]])
                    except:  # It is a string
                        csv_data[col] = rows[pos[col]]
            imported_data.append(csv_data)

    return imported_data


def read_var_csv(filename, datatype, truedataindex):
    """
    Parameters
    ----------
    filename : str
        Name of the CSV file.

    datatype : list
        List of data types as strings.

    truedataindex : list
        List of indices where the "TRUEDATA" has been extracted.

    Returns
    -------
    imported_var : list
        List of variances.
    """

    df = pd.read_csv(filename)  # Read the file

    imported_var = []  # Initialize the 2D list of csv data
    tlength = len(truedataindex)
    dnumber = len(datatype)

    if df.columns[0] == 'header_both':  # csv file has column and row headers
        pos = [None] * dnumber
        for col in range(dnumber):
            # find index of data type in csv file header
            pos[col] = df.columns.get_loc(datatype[col])
        for t in truedataindex:
            row = df[df['header_both'] == t]  # pick row
            row = row.values[0]  # select the values of the dataframe
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = row[pos[col]]
                try:  # Making a float
                    csv_data[2*col+1] = float(row[pos[col]]+1)
                except:  # It is a string
                    csv_data[2*col+1] = row[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)
    else:  # No row headers (the rows in the csv file must correspond to the order in truedataindex)
        if tlength == df.shape[0]:  # File has column headers
            pos = [None] * dnumber
            for col in range(dnumber):
                # Find index of datatype in csv file header
                pos[col] = df.columns.get_loc(datatype[col])
        # File has no column headers (columns must correspond to the order in datatype)
        elif tlength == df.shape[0]+1:
            # First row has been misinterpreted as header, so we read first row again:
            temp = pd.read_csv(filename, header=None, nrows=1).values[0]
            # Make sure the string input is lowercase
            temp[0::2] = [x.lower() for x in temp[0::2]]
            # Assume the data is in the correct order
            pos = list(range(0, df.shape[1], 2))
            csv_data = [None] * len(temp)
            for col in range(dnumber):
                csv_data[2 * col] = temp[2 * col]
                try:  # Making a float
                    csv_data[2*col+1] = float(temp[2*col+1])
                except:  # It is a string
                    csv_data[2*col+1] = temp[2*col+1]
            imported_var.append(csv_data)

        for rows in df.values:
            csv_data = [None] * 2 * dnumber
            for col in range(dnumber):
                csv_data[2*col] = rows[2*col]
                try:  # Making a float
                    csv_data[2*col+1] = float(rows[pos[col]+1])
                except:  # It is a string
                    csv_data[2*col+1] = rows[pos[col]+1]
            # Make sure the string input is lowercase
            csv_data[0::2] = [x.lower() for x in csv_data[0::2]]
            imported_var.append(csv_data)

    return imported_var
