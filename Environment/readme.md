# Develop Language
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Information
**Creator** : Seungyeon Lee \
**Date** : 2024-12-27

# 현재 상황
1. **TTT_Environment.py** 는 완전하게 작동하는 환경입니다.
2. **TTT_Environment2.py** 는 State 클래스를 구현한 상태입니다. reward를 주는 등 Environment의 역할을 다하지 못하는 상태입니다.

- 2의 경우, 1) environment 클래스를 따로 구현하는 방법과 2) main에서 environment의 역할을 대신하도록 구현하는 방법을 선택해야 합니다.
- 현재 1번 환경의 속도가 조금 더 빠릅니다.
- 2번 환경을 굳이 수정해서 업로드한 이유는 게임트리와의 상호작용이 더 좋을 것으로 예상되기 때문입니다. 우선적으로 1번 환경을 사용하되, 게임트리에서 너무 복잡해진다면 2번 환경을 사용하는 방법을 고려해볼 필요가 있습니다.