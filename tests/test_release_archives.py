from __future__ import annotations
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile
import pytest

spec=importlib.util.spec_from_file_location('package_release',Path(__file__).parents[1]/'scripts/package_release.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_round_trip_and_source_preservation(tmp_path):
    src=tmp_path/'best.pt';src.write_bytes(b'checkpoint bytes')
    recipe={'id':'example','files':[{'source':str(src),'name':'run/best.pt'}]}
    out=tmp_path/'release';out.mkdir()
    receipt=m.package(recipe,out)
    m.verify_extract(out/receipt['zip'],tmp_path/'unpacked',receipt['zip_sha256'])
    assert (tmp_path/'unpacked/run/best.pt').read_bytes()==src.read_bytes()
    assert src.read_bytes()==b'checkpoint bytes'
    assert m.package(recipe,out)==receipt


@pytest.mark.parametrize('name',['../escape','/absolute','C:/windows','x\\escape','.env','run/.git/config'])
def test_rejects_unsafe_members(name):
    with pytest.raises(ValueError):m.safe_name(name)


def test_active_pipeline_not_packaged(tmp_path):
    state=tmp_path/'state.json';state.write_text('{"status":"running"}')
    recipe={'id':'active','requires_complete':[str(state)]}
    assert not m.eligible(recipe)
    with pytest.raises(ValueError):m.package(recipe,tmp_path)
    assert not list(tmp_path.glob('*.zip'))


def test_tar_prefix_filter_and_zip64(tmp_path):
    archive=tmp_path/'old.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['runA/best.pt','runB/best.pt']:
            data=name.encode();info=tarfile.TarInfo(name);info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    recipe={'id':'from-tar','tar_trees':[{'source':str(archive),'member_prefix':'runA','prefix':'run'}]}
    receipt=m.package(recipe,tmp_path/'release')
    with zipfile.ZipFile(tmp_path/'release'/receipt['zip']) as z:
        assert z.read('run/best.pt')==b'runA/best.pt'
        assert not any('runB' in name for name in z.namelist())


def test_extract_rejects_unmanifested_file(tmp_path):
    p=tmp_path/'malformed.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('FILES.json','[]');z.writestr('extra.txt','unexpected')
    with pytest.raises(ValueError,match='Unmanifested'):
        m.verify_extract(p,tmp_path/'extract')
