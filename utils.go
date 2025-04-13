package einox

import (
	"fmt"
	"strings"

	"github.com/cloudwego/eino/schema"
	"github.com/sashabaranov/go-openai" // 确保导入 go-openai 包
)

// convertChatRequestToSchemaMessages 将ChatRequest中的消息转换为schema.Message格式
func convertChatRequestToSchemaMessages(req ChatRequest) []*schema.Message {
	schemaMessages := make([]*schema.Message, len(req.Messages))
	for i, msg := range req.Messages {
		// 创建基本消息结构
		schemaMsg := &schema.Message{
			Role:       schema.RoleType(msg.Role),
			Name:       msg.Name,
			ToolCallID: msg.ToolCallID, // 主要用于 'tool' 角色的消息
			//TODO 待完善
			//ToolCalls: convertToolCalls(msg.ToolCalls),
		}

		// 处理内容 - 根据是否有多模态内容决定使用Content还是MultiContent
		if len(msg.MultiContent) > 0 {
			// 处理多模态内容
			multiContent := make([]schema.ChatMessagePart, len(msg.MultiContent))
			for j, part := range msg.MultiContent {
				chatPart := schema.ChatMessagePart{
					Type: schema.ChatMessagePartType(part.Type),
					Text: part.Text,
				}

				// 处理不同类型的媒体URL
				switch chatPart.Type {
				case schema.ChatMessagePartTypeImageURL:
					// 处理图片URL
					if part.ImageURL != nil {
						// 判断是否为URL格式，如果是则转换为BASE64
						if isURL(part.ImageURL.URL) {
							// 转换图片URL为BASE64
							base64Data, mimeType, err := convertImageURLToBase64(part.ImageURL.URL)
							if err != nil {
								// 记录错误但继续使用原URL结构
								fmt.Printf("转换图片URL到BASE64失败: %v\n", err)
								// 保留原始 ImageURL 结构（如果转换失败）
								chatPart.ImageURL = &schema.ChatMessageImageURL{
									URL:    part.ImageURL.URL,
									Detail: schema.ImageURLDetail(part.ImageURL.Detail),
									// MIMEType 可能未知
								}
							} else {
								// 使用转换后的BASE64数据
								chatPart.ImageURL = &schema.ChatMessageImageURL{
									URL:      base64Data,
									Detail:   schema.ImageURLDetail(part.ImageURL.Detail),
									MIMEType: mimeType,
								}
							}
						} else {
							// 默认处理方式，可能已经是BASE64数据
							chatPart.ImageURL = &schema.ChatMessageImageURL{
								URL:      part.ImageURL.URL,
								Detail:   schema.ImageURLDetail(part.ImageURL.Detail),
								MIMEType: detectMIMEType(part.ImageURL.URL),
							}
						}
					}
				case schema.ChatMessagePartTypeAudioURL:
					// 处理音频URL (如果API支持)
					if part.ImageURL != nil { // 临时使用ImageURL字段
						chatPart.AudioURL = &schema.ChatMessageAudioURL{
							URL:      part.ImageURL.URL,
							MIMEType: "audio/mp3", // 默认MIME类型
						}
					}
				case schema.ChatMessagePartTypeVideoURL:
					// 处理视频URL (如果API支持)
					if part.ImageURL != nil { // 临时使用ImageURL字段
						chatPart.VideoURL = &schema.ChatMessageVideoURL{
							URL:      part.ImageURL.URL,
							MIMEType: "video/mp4", // 默认MIME类型
						}
					}
				case schema.ChatMessagePartTypeFileURL:
					// 处理文件URL (如果API支持)
					if part.ImageURL != nil { // 临时使用ImageURL字段
						chatPart.FileURL = &schema.ChatMessageFileURL{
							URL:      part.ImageURL.URL,
							MIMEType: "application/pdf", // 默认MIME类型
							Name:     "file.pdf",        // 默认文件名 TODO 待完善
						}
					}
				}

				multiContent[j] = chatPart
			}
			schemaMsg.MultiContent = multiContent
		} else {
			// 使用普通文本内容
			schemaMsg.Content = msg.Content
		}

		// --- 处理 Assistant 的 ToolCalls ---
		if msg.Role == openai.ChatMessageRoleAssistant && len(msg.ToolCalls) > 0 {
			schemaToolCalls := make([]schema.ToolCall, 0, len(msg.ToolCalls)) // 初始化为空切片
			for _, tc := range msg.ToolCalls {
				// 仅转换 function 类型的 tool call
				if tc.Type == openai.ToolTypeFunction {
					schemaToolCalls = append(schemaToolCalls, schema.ToolCall{
						ID:   tc.ID,
						Type: string(tc.Type), // 转换为 "function" 字符串
						Function: schema.FunctionCall{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
						// Index 字段通常在非流式请求的转换中不需要设置
					})
				} else {
					fmt.Printf("[DEBUG-UTILS] 警告: 跳过非 function 类型的工具调用: Type=%s, ID=%s\n", tc.Type, tc.ID)
				}
			}
			// 只有当确实转换了 tool call 时才赋值
			if len(schemaToolCalls) > 0 {
				schemaMsg.ToolCalls = schemaToolCalls
			}
		}
		// --- 结束处理 ToolCalls ---

		// 如果存在额外数据，添加到Extra字段
		if req.Extra != nil && len(req.Extra) > 0 {
			schemaMsg.Extra = req.Extra
		}

		// 保存转换后的消息
		schemaMessages[i] = schemaMsg
	}

	return schemaMessages
}

// isURL (需要实现或确保存在) - 简单实现
func isURL(s string) bool {
	return strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://")
}

// convertImageURLToBase64 (需要实现或确保存在) - 占位符实现
// 在实际场景中，这里需要获取URL内容、编码为Base64并检测MIME类型
func convertImageURLToBase64(url string) (string, string, error) {
	fmt.Printf("[DEBUG-UTILS] 占位符: 需要实现将 URL %s 转换为 base64\n", url)
	// 暂时返回错误以模拟原始逻辑流程（打印错误但不中断）
	return "", "", fmt.Errorf("convertImageURLToBase64 未实现")
}

// detectMIMEType 根据URL或数据检测MIME类型
func detectMIMEType(urlOrData string) string {
	// 简单检测MIME类型
	if strings.HasPrefix(urlOrData, "data:image/png;") {
		return "image/png"
	} else if strings.HasPrefix(urlOrData, "data:image/jpeg;") {
		return "image/jpeg"
	} else if strings.HasPrefix(urlOrData, "data:image/gif;") {
		return "image/gif"
	} else if strings.HasPrefix(urlOrData, "data:image/webp;") {
		return "image/webp"
	} else if strings.HasPrefix(urlOrData, "data:image/") {
		return "image/png" // 默认图片类型
	} else if strings.HasSuffix(urlOrData, ".png") {
		return "image/png"
	} else if strings.HasSuffix(urlOrData, ".jpg") || strings.HasSuffix(urlOrData, ".jpeg") {
		return "image/jpeg"
	} else if strings.HasSuffix(urlOrData, ".gif") {
		return "image/gif"
	} else if strings.HasSuffix(urlOrData, ".webp") {
		return "image/webp"
	}

	return "image/jpeg" // 默认MIME类型
}
