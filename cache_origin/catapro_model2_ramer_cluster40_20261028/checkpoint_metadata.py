"""Read only the torch ZIP pickle metadata; never deserialize or read tensor storage."""
import collections
import io
import json
import pickle
import sys
import zipfile


class Storage:
    def __init__(self, name):
        self.name = name


def rebuild(storage, offset, size, stride, *args):
    return dict(shape=list(size), stride=list(stride), storage=storage,
                offset=offset)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return rebuild
        if module == 'torch' and name.endswith('Storage'):
            return Storage(name)
        raise ValueError('Unsupported pickle global: ' + module + '.' + name)

    def persistent_load(self, value):
        if value[0] != 'storage' or not isinstance(value[1], Storage):
            raise ValueError('Unsupported persistent storage')
        return dict(dtype=value[1].name, key=value[2], device=value[3], size=value[4])


def main():
    with zipfile.ZipFile(sys.argv[1]) as archive:
        candidates = [p for p in archive.namelist() if p.endswith('/data.pkl')]
        if len(candidates) != 1:
            raise ValueError('Expected a single torch ZIP data.pkl')
        info = archive.getinfo(candidates[0])
        if info.file_size > 16 * 1024 * 1024:
            raise ValueError('Unexpectedly large checkpoint metadata')
        state = MetadataUnpickler(io.BytesIO(archive.read(candidates[0]))).load()
    if not isinstance(state, dict) or not state or not all(isinstance(v, dict) and 'shape' in v for v in state.values()):
        raise ValueError('Checkpoint is not a plain state_dict')
    print(json.dumps(dict(path=sys.argv[1], tensor_count=len(state),
                         metadata_bytes=info.file_size, tensor_storage_read=False,
                         tensors=state)))


if __name__ == '__main__':
    main()
