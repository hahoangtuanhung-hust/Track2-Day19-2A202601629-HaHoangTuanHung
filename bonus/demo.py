import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent

def main():
    print("Initializing Hybrid Memory Agent...")
    agent = HybridMemoryAgent()
    
    user_id = "u_001"
    
    print("Loading episodic memories...")
    memories = [
        "Kubernetes (K8s) là một hệ thống mã nguồn mở tự động hóa việc quản lý, mở rộng và triển khai ứng dụng dưới dạng container.",
        "Docker là nền tảng phần mềm cho phép tạo, kiểm thử và triển khai ứng dụng nhanh chóng bằng container.",
        "Phương pháp tự động mở rộng hạ tầng (autoscaling) giúp hệ thống tự động thêm bớt server dựa trên lưu lượng thực tế.",
        "Cloud security (bảo mật đám mây) yêu cầu mã hóa dữ liệu tại chỗ, cấu hình IAM nghiêm ngặt và giám sát liên tục.",
        "Ghi chú: Cần đọc thêm tài liệu về tối ưu chi phí hạ tầng AWS vào tuần sau."
    ]
    
    for m in memories:
        agent.remember(m, user_id=user_id)
        
    print(f"Loaded {len(memories)} memories for user {user_id}.\n")
    
    queries = [
        "Tôi đã đọc gì về Kubernetes và Docker?",
        "Đề xuất bài đọc phù hợp với sở thích của tôi",
        "Gần đây tôi đang tìm kiếm chủ đề gì?",
        "Tài liệu tự động co giãn hạ tầng đám mây",
        "Tóm tắt các ghi chú về bảo mật cloud"
    ]
    
    print("=" * 50)
    for i, query in enumerate(queries, 1):
        print(f"\nQUERY {i}: {query}")
        context = agent.recall(query, user_id=user_id, top_k=2)
        print(context)
    print("\n" + "=" * 50)
    print("Demo completed successfully.")

if __name__ == "__main__":
    main()
