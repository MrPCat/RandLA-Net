import numpy as np
import sys

# Define PLY types as dictionary for reference
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format mapping for ASCII and binary formats
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    """
    Parses the header of a PLY file to extract properties and number of points.
    """
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties

def parse_mesh_header(plyfile, ext):
    """
    Parses the header of a mesh-format PLY file to extract vertex and face properties.
    """
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'face':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property: ' + line)

    return num_points, num_faces, vertex_properties

def read_ply(filename, triangular_mesh=False):
    """
    Reads a PLY file and returns the data.

    - If `triangular_mesh` is True, the file is expected to be a mesh with vertex and face data.
    - Otherwise, it reads point cloud data.

    Returns a numpy structured array with the data.
    """
    with open(filename, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start with the word "ply"')

        # Get the format type: binary, ASCII, etc.
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        ext = valid_formats[fmt]

        if triangular_mesh:
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            face_properties = [('k', ext + 'u1'), ('v1', ext + 'i4'), ('v2', ext + 'i4'), ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            return [vertex_data, faces]

        else:
            num_points, properties = parse_header(plyfile, ext)
            return np.fromfile(plyfile, dtype=properties, count=num_points)

def header_properties(field_list, field_names):
    """
    Generates the header information for a PLY file, based on the fields to be written.
    """
    lines = []
    lines.append('element vertex %d' % field_list[0].shape[0])

    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines

def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Writes data to a PLY file.

    Supports saving point cloud data (with multiple fields) and mesh data (with vertex and triangular face data).
    """
    field_list = list(field_list) if isinstance(field_list, (list, tuple)) else [field_list]
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('Fields have more than 2 dimensions')
            return False

    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('Wrong field dimensions')
        return False

    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print('Wrong number of field names')
        return False

    if not filename.endswith('.ply'):
        filename += '.ply'

    with open(filename, 'w') as plyfile:
        header = ['ply', 'format binary_' + sys.byteorder + '_endian 1.0']
        header.extend(header_properties(field_list, field_names))

        if triangular_faces is not None:
            header.append(f'element face {triangular_faces.shape[0]}')
            header.append('property list uchar int vertex_indices')

        header.append('end_header')
        for line in header:
            plyfile.write("%s\n" % line)

    with open(filename, 'ab') as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list.append((field_names[i], field.dtype.str))
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True

def describe_element(name, df):
    """
    Takes a dataframe and returns a list of descriptions of each element, suitable for creating a PLY header.
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = [f'element {name} {len(df)}']

    if name == 'face':
        element.append("property list uchar int points_indices")
    else:
        for i in range(len(df.columns)):
            f = property_formats[str(df.dtypes[i])[0]]
            element.append(f'property {f} {df.columns.values[i]}')

    return element
