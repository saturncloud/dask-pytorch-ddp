from unittest.mock import Mock, patch, ANY


from dask_pytorch.data import S3ImageFolder


def test_image_folder_constructor():
    fake_file_list = ["d/a.jpg", "c/b.jpg"]
    with patch("dask_pytorch.data._list_all_files", return_value=fake_file_list):
        fake_transform = Mock()
        fake_target_transform = Mock()
        folder = S3ImageFolder(
            "fake-bucket",
            "fake-prefix/fake-prefix",
            fake_transform,
            fake_target_transform,
        )
    assert folder.all_files == fake_file_list
    assert folder.classes == ["c", "d"]
    assert folder.class_to_idx == {"c": 0, "d": 1}
    assert folder.transform == fake_transform
    assert folder.target_transform == fake_target_transform


def test_image_folder_len():
    fake_file_list = ["d/a.jpg", "c/b.jpg"]
    with patch("dask_pytorch.data._list_all_files", return_value=fake_file_list):
        folder = S3ImageFolder("fake-bucket", "fake-prefix/fake-prefix")
    assert len(folder) == 2


def test_image_folder_getitem():
    fake_file_list = ["d/a.jpg", "c/b.jpg"]
    with patch("dask_pytorch.data._list_all_files", return_value=fake_file_list):
        folder = S3ImageFolder("fake-bucket", "fake-prefix/fake-prefix")
    with patch("dask_pytorch.data._read_s3_fileobj") as read_s3_fileobj, patch(
        "dask_pytorch.data._load_image_obj"
    ) as load_image_obj:

        read_s3_fileobj.return_value = Mock()
        load_image_obj.return_value = Mock()
        val, label = folder[0]
        read_s3_fileobj.assert_called_once_with("fake-bucket", fake_file_list[0], ANY)
        load_image_obj.assert_called_once_with(read_s3_fileobj())
        assert val == load_image_obj.return_value
        assert label == 1
