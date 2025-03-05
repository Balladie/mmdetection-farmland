import os
import argparse
from pathlib import Path

def compare_directories(inputs_path, outputs_path):
    """
    입력 폴더와 출력 폴더의 파일들을 비교합니다.
    파일명의 확장자를 제외하고 비교하며, 차이가 있는 경우 해당 정보를 출력합니다.
    
    Args:
        inputs_path (str): 입력 폴더 경로
        outputs_path (str): 출력 폴더 경로 (preds 폴더의 상위 경로)
    """
    # 입력 경로가 존재하지 않는 경우 처리
    if not os.path.exists(inputs_path):
        print(f"입력 폴더가 존재하지 않습니다: {inputs_path}")
        return
        
    # 출력 경로 확인
    if not os.path.exists(outputs_path):
        print(f"출력 폴더가 존재하지 않습니다: {outputs_path}")
        return

    # 모든 하위 폴더를 순회하면서 파일 검사
    for root, _, files in os.walk(inputs_path):
        # 현재 검사중인 input 폴더의 상대 경로 구하기
        rel_path = os.path.relpath(root, inputs_path)
        output_base_dir = os.path.join(outputs_path, rel_path)
        output_preds_dir = os.path.join(output_base_dir, 'preds')
        
        # input 파일들의 이름만 추출 (확장자 제외)
        input_files = set()
        for f in files:
            input_files.add(os.path.splitext(f)[0])
        
        # 파일이 없는 경우 다음 폴더로 진행
        if not input_files:
            continue

        # preds 폴더가 없는 경우
        if not os.path.exists(output_preds_dir):
            print(f"폴더 경로: {root}")
            print(f"상태: FAIL | Input 파일 수: {len(input_files)} | Output 파일 수: 0 | 사유: preds 폴더 없음")
            continue
            
        # output 파일들의 이름만 추출 (확장자 제외)
        output_files = set()
        for f in os.listdir(output_preds_dir):
            if os.path.isfile(os.path.join(output_preds_dir, f)):
                output_files.add(os.path.splitext(f)[0])
        
        print(f"폴더 경로: {root}")
        
        # 파일 수 비교
        if input_files != output_files:
            print(f"상태: FAIL | Input 파일 수: {len(input_files)} | Output 파일 수: {len(output_files)}")
            # input에만 있는 파일 출력
            missing_in_output = input_files - output_files
            if missing_in_output:
                print("Output에 없는 파일들:")
                for f in sorted(missing_in_output):
                    print(f"  - {f}")
            
            # output에만 있는 파일 출력
            extra_in_output = output_files - input_files
            if extra_in_output:
                print("Input에 없는 파일들:")
                for f in sorted(extra_in_output):
                    print(f"  - {f}")
        else:
            print(f"상태: PASS | Input 파일 수: {len(input_files)} | Output 파일 수: {len(output_files)}")

def main():
    """
    메인 함수: 커맨드 라인 인자를 파싱하고 디렉토리 비교를 실행합니다.
    출력 폴더는 지정된 경로 아래의 'preds' 폴더에서 파일을 찾습니다.
    """
    # 커맨드 라인 인자 파서 설정
    parser = argparse.ArgumentParser(
        description='입력/출력 폴더의 파일 개수를 비교합니다.\n'
                   '출력 파일은 output-dir/preds 경로에서 찾습니다.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='입력 폴더 경로'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='출력 폴더의 상위 경로 (preds 폴더가 위치한 경로)'
    )
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 디렉토리 비교 실행
    compare_directories(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
