import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import builtins

from app import find_best_tool, TOOLS, all_tools

def test_find_best_tool_match_high_confidence():
    user_query = "I want an ascii art effect"
    tool, confidence = find_best_tool(user_query, TOOLS)
    assert tool['name'] == "ASCII-fy"
    assert confidence > 0.2

def test_find_best_tool_no_input():
    tool, confidence = find_best_tool("", TOOLS)
    assert tool is None
    assert confidence == 0.0

@patch('builtins.input', side_effect=["non_existent_image.jpg", "q"])
@patch('cv2.imread', return_value=None)
def test_all_tools_file_not_found(mock_imread, mock_input):
    # all_tools should loop and handle missing file gracefully, then user quits
    with pytest.raises(StopIteration):  # Because "q" ends input unexpectedly here
        all_tools("get-aesthetic-photo")

@patch('builtins.input', side_effect=["test_image.jpg"])
@patch('cv2.imread')
@patch('cv2.imwrite')
@patch('os.path.basename', return_value="test_image.jpg")
@patch('subprocess.call')
@patch('platform.system', return_value='Linux')
def test_all_tools_process_and_save(mock_platform, mock_subprocess, mock_basename, mock_imwrite, mock_imread, mock_input):
    test_img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    mock_imread.return_value = test_img
    all_tools("get-aesthetic-photo")
    mock_imwrite.assert_called_once()
    mock_subprocess.assert_called_once()

